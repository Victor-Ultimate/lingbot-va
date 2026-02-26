# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Train with trajectory GT: end-effector trajectory = tcp_pose_wrt_base (7) + base_action (2).
Data from H5 via TrajectoryChunkDataset (LeRobot traj dataset).
"""
import argparse
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model,
    init_distributed,
    dist_mean,
    dist_max,
)
from einops import rearrange
from modules.utils import load_transformer
from utils import (
    init_logger,
    logger,
    get_mesh_id,
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler,
)

from dataset import TrajectoryChunkDataset
import gc


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            wandb.login(host=os.environ.get("WANDB_BASE_URL", ""), key=os.environ.get("WANDB_API_KEY", ""))
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ.get("WANDB_TEAM_NAME", ""),
                project=os.getenv("WANDB_PROJECT", "va_robotwin_trajectory"),
                config=dict(config),
                mode="offline",
                name=getattr(config, "run_name", "trajectory_train"),
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.latent_loss_weight = getattr(config, "latent_loss_weight", 0.0)

        logger.info("Loading transformer...")
        if hasattr(config, "resume_from") and config.resume_from:
            transformer_path = os.path.join(config.resume_from, "transformer")
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, "transformer")

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps),
        )

        logger.info("Setting up trajectory dataset (H5)...")
        train_dataset = TrajectoryChunkDataset(
            config.h5_dataset_path,
            config,
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42,
            )
            if config.world_size > 1
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=getattr(config, "load_worker", 4),
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.0):
        B, C, F, H, W = latent.shape
        timestep_ids = sample_timestep_id(
            batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps
        )
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=F,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        latent_dict = self._add_noise(
            latent=batch_dict["latents"],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
        )
        action_dict = self._add_noise(
            latent=batch_dict["actions"],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict["actions_mask"],
            action_mode=True,
            noisy_cond_prob=0.0,
        )
        latent_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["actions_mask"] = batch_dict["actions_mask"]

        input_dict = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": torch.randint(1, 5, (1,)).item(),
            "window_size": torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, batch_dict):
        for key, value in batch_dict.items():
            batch_dict[key] = value.to(self.device)
        return batch_dict

    def compute_loss(self, input_dict, pred):
        latent_pred, action_pred = pred
        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        Bn, Fn = input_dict["latent_dict"]["timesteps"].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(Bn, Fn)

        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict["action_dict"]["actions_mask"].float()
        action_loss = action_loss.permute(0, 2, 3, 4, 1)
        action_mask = input_dict["action_dict"]["actions_mask"].float().permute(0, 2, 3, 4, 1)
        action_loss = action_loss.flatten(0, 1).flatten(1)
        action_mask = action_mask.flatten(0, 1).flatten(1)
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        scale = 1.0 / self.gradient_accumulation_steps
        total_latent = self.latent_loss_weight * latent_loss * scale
        total_action = action_loss * scale
        return total_latent, total_action

    def train_epoch(self):
        self.transformer.train()
        progress_bar = tqdm(
            total=len(self.train_loader),
            desc="Training (trajectory)",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self.convert_input_format(batch)
            input_dict = self._prepare_input_dict(batch)

            should_sync = (
                (batch_idx + 1) % self.gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(self.train_loader)
            )

            if not should_sync:
                self.transformer.set_requires_gradient_sync(False)
            else:
                self.transformer.set_requires_gradient_sync(True)

            output = self.transformer(input_dict, train_mode=True)
            latent_loss, action_loss = self.compute_loss(input_dict, output)
            loss = latent_loss + action_loss

            loss.backward()
            accumulated_latent_losses.append(latent_loss.detach())
            accumulated_action_losses.append(action_loss.detach())

            if should_sync:
                total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                lr = self.lr_scheduler.get_last_lr()[0]
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                accumulated_latent_losses = []
                accumulated_action_losses = []

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        "latent_loss": f"{latent_loss_show:.4f}",
                        "traj_loss": f"{action_loss_show:.4f}",
                        "step": self.step,
                        "grad_norm": f"{total_norm.item():.2f}",
                        "lr": f"{lr:.2e}",
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            "loss_metrics/global_avg_video_loss": latent_loss_show,
                            "loss_metrics/global_avg_trajectory_loss": action_loss_show,
                            "loss_metrics/global_max_video_loss": max_latent_loss_show,
                            "loss_metrics/global_max_trajectory_loss": max_action_loss_show,
                            "grad_norm": total_norm.item(),
                            "lr": lr,
                        }, step=self.step)
                self.step += 1
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

        progress_bar.close()

    def save_checkpoint(self):
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving transformer to {transformer_dir}")
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop("_name_or_path", None)
                with open(config_file, "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"Checkpoint saved successfully at step {self.step}")

            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    def train(self):
        logger.info("Starting trajectory training...")
        while self.step < self.config.num_steps:
            self.train_epoch()
            if dist.is_initialized():
                dist.barrier()
        logger.info("Training completed!")


def run(args):
    config = VA_CONFIGS[args.config_name]
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.h5_path is not None:
        config.h5_dataset_path = args.h5_path

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"H5 dataset path: {config.h5_dataset_path}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train WAN with trajectory GT (tcp_pose + base_action)")
    parser.add_argument(
        "--config-name",
        type=str,
        default="robotwin_trajectory_train",
        help="Config name",
    )
    parser.add_argument("--save-root", type=str, default=None, help="Root directory for checkpoints")
    parser.add_argument("--h5-path", type=str, default=None, help="Override H5 dataset path (file or dir)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
