# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
LeRobot-style Dataset that reads from H5 trajectory files.

H5 structure per trajectory (traj_XXX):
  obs/
    agent/     : qpos (201, 12), qvel (201, 12)
    extra/     : tcp_pose_wrt_base (201, 7), obj_pose_wrt_base (201, 7),
                 goal_pos_wrt_base (201, 3), is_grasped (201,)
    sensor_param/, sensor_data/  : optional cameras (fetch_head, fetch_hand)
  actions      : (200, 13)
  terminated, truncated, success, fail, rewards : (200,)

Each item is indexed by (traj_id, frame_idx) with frame_idx in [0, 199].
Output format follows LeRobot: action + observation (state, extra required; images optional).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import numpy as np
import torch
import h5py


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _discover_traj_keys(h5: h5py.File) -> List[str]:
    """Return sorted list of group names like 'traj_0', 'traj_1', ..."""
    keys = [k for k in h5.keys() if isinstance(h5[k], h5py.Group) and re.match(r"traj_\d+", k)]
    return sorted(keys, key=lambda x: int(x.split("_")[1]))


def _traj_key_to_episode_index(traj_key: str) -> int:
    return int(traj_key.split("_")[1])


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class LeRobotTrajDataset(torch.utils.data.Dataset):
    """
    Reads H5 trajectory file(s) and exposes LeRobot-style items:
    - action (required)
    - observation.state (qpos), observation.extra (tcp_pose_wrt_base, obj_pose_wrt_base, goal_pos_wrt_base, is_grasped)
    - optionally observation.agent.qvel, observation.images.*
    """

    # Keys under obs/extra that must be kept
    EXTRA_KEYS = ("tcp_pose_wrt_base", "obj_pose_wrt_base", "goal_pos_wrt_base", "is_grasped")

    def __init__(
        self,
        h5_path: Union[str, Path],
        *,
        keep_images: bool = True,
        keep_qvel: bool = True,
        frame_index_range: Optional[Tuple[int, int]] = None,
        traj_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            h5_path: Path to a single .h5 file or directory of .h5 files.
            keep_images: If True, load and return observation.images.fetch_head.rgb, fetch_hand.rgb.
            keep_qvel: If True, return observation.agent.qvel.
            frame_index_range: Optional (start, end) to use only frames in [start, end) per traj (0-indexed, max 200).
            traj_keys: Optional list of trajectory group names to use (e.g. ['traj_0','traj_1']). If None, use all traj_*.
        """
        self.h5_path = Path(h5_path)
        self.keep_images = keep_images
        self.keep_qvel = keep_qvel
        self.frame_index_range = frame_index_range
        self._h5_files: List[Path] = []
        self._index: List[Tuple[Path, str, int]] = []  # (file, traj_key, frame_idx)

        self._build_index(traj_keys)

    def _collect_h5_files(self) -> List[Path]:
        if self.h5_path.is_file() and self.h5_path.suffix.lower() in (".h5", ".hdf5"):
            return [self.h5_path]
        if self.h5_path.is_dir():
            files = sorted(self.h5_path.glob("*.h5")) + sorted(self.h5_path.glob("*.hdf5"))
            return files
        return []

    def _build_index(self, traj_keys: Optional[List[str]]) -> None:
        self._h5_files = self._collect_h5_files()
        if not self._h5_files:
            raise FileNotFoundError(f"No H5 file(s) found at {self.h5_path}")

        start_f, end_f = self.frame_index_range or (0, 200)
        start_f = max(0, start_f)
        end_f = min(200, end_f)

        for fpath in self._h5_files:
            with h5py.File(fpath, "r") as h5:
                keys = traj_keys if traj_keys is not None else _discover_traj_keys(h5)
                for traj_key in keys:
                    if traj_key not in h5:
                        continue
                    g = h5[traj_key]
                    if "actions" not in g:
                        continue
                    n_actions = g["actions"].shape[0]
                    for fi in range(start_f, min(end_f, n_actions)):
                        self._index.append((fpath, traj_key, fi))

        if not self._index:
            raise ValueError(f"No (traj, frame) pairs found under {self.h5_path}")

    def __len__(self) -> int:
        return len(self._index)

    def _load_traj_group(self, fpath: Path, traj_key: str) -> h5py.Group:
        # Keep file open only during __getitem__ to avoid holding many handles
        return h5py.File(fpath, "r")[traj_key]

    def _get_obs_extra(self, g: h5py.Group, frame_idx: int) -> Dict[str, np.ndarray]:
        out = {}
        extra = g["obs/extra"]
        for k in self.EXTRA_KEYS:
            if k in extra:
                d = extra[k]
                if d.ndim == 1:
                    out[k] = np.array(d[frame_idx], dtype=np.float32) if d.dtype != np.bool_ else np.array(d[frame_idx])
                else:
                    out[k] = np.array(d[frame_idx], dtype=np.float32)
        return out

    def _get_obs_agent(self, g: h5py.Group, frame_idx: int) -> Dict[str, np.ndarray]:
        out = {}
        agent = g["obs/agent"]
        if "qpos" in agent:
            out["qpos"] = np.array(agent["qpos"][frame_idx], dtype=np.float32)
        if self.keep_qvel and "qvel" in agent:
            out["qvel"] = np.array(agent["qvel"][frame_idx], dtype=np.float32)
        return out

    def _get_obs_images(self, g: h5py.Group, frame_idx: int) -> Dict[str, np.ndarray]:
        out = {}
        if not self.keep_images or "obs/sensor_data" not in g:
            return out
        for cam_name in ("fetch_head", "fetch_hand"):
            cam_path = f"obs/sensor_data/{cam_name}"
            if cam_path not in g:
                continue
            cam = g[cam_path]
            if "rgb" in cam:
                out[f"{cam_name}.rgb"] = np.array(cam["rgb"][frame_idx], dtype=np.uint8)
        return out

    def __getitem__(self, idx: int) -> Dict:
        fpath, traj_key, frame_idx = self._index[idx]
        with h5py.File(fpath, "r") as h5:
            g = h5[traj_key]

            # Action (required for LeRobot)
            action = np.array(g["actions"][frame_idx], dtype=np.float32)

            # Observation: state (qpos) + extra (required)
            obs_agent = self._get_obs_agent(g, frame_idx)
            obs_extra = self._get_obs_extra(g, frame_idx)
            obs_images = self._get_obs_images(g, frame_idx)

        # LeRobot-style nested observation
        observation = {
            "state": obs_agent.get("qpos", np.zeros(12, dtype=np.float32)),
            "extra": obs_extra,
        }
        if obs_agent.get("qvel") is not None:
            observation["agent"] = {"qpos": obs_agent["qpos"], "qvel": obs_agent["qvel"]}
        else:
            observation["agent"] = {"qpos": obs_agent["qpos"]}
        if obs_images:
            observation["images"] = obs_images

        episode_index = _traj_key_to_episode_index(traj_key)

        return {
            "observation": observation,
            "action": action,
            "episode_index": episode_index,
            "frame_index": frame_idx,
            "timestamp": float(frame_idx),
            "traj_key": traj_key,
        }


# Fetch action layout: indices 0-6 arm, 7 gripper, 8-10 body, 11-12 base (forward_vel, rotation_vel)
TRAJECTORY_DIM = 9  # tcp_pose_wrt_base (7) + base_action (2)
BASE_ACTION_SLICE = (11, 13)  # action[:, 11:13] for base


class MultiLeRobotTrajDataset(torch.utils.data.Dataset):
    """
    Unified dataset: accepts a single H5 path or a directory (multiple H5 files).
    - mode="frame": per-frame LeRobot-style (observation + action, obs/extra, optional images/qvel).
    - mode="chunk": chunk-level trajectory training (latents, text_emb, actions as trajectory GT,
      actions_mask). GT = tcp_pose_wrt_base (7) + base_action (2), padded to action_dim.
    H5 dir: when h5_path is a directory, all *.h5 / *.hdf5 under it are used.
    """

    EXTRA_KEYS = ("tcp_pose_wrt_base", "obj_pose_wrt_base", "goal_pos_wrt_base", "is_grasped")

    def __init__(
        self,
        h5_path: Union[str, Path],
        config=None,
        *,
        mode: str = "chunk",
        # frame mode
        keep_images: bool = True,
        keep_qvel: bool = True,
        frame_index_range: Optional[Tuple[int, int]] = None,
        # chunk mode (from config if not set)
        frame_chunk_size: Optional[int] = None,
        action_per_frame: Optional[int] = None,
        action_dim: Optional[int] = None,
        traj_keys: Optional[List[str]] = None,
    ):
        self.h5_path = Path(h5_path)
        self.config = config or type("Config", (), {})()
        self.mode = mode
        self.keep_images = keep_images
        self.keep_qvel = keep_qvel
        self.frame_index_range = frame_index_range
        # Chunk params (used when mode=="chunk")
        self.frame_chunk_size = frame_chunk_size or getattr(self.config, "frame_chunk_size", 2)
        self.action_per_frame = action_per_frame or getattr(self.config, "action_per_frame", 16)
        self.action_dim = action_dim or getattr(self.config, "action_dim", 30)
        self.chunk_len = self.frame_chunk_size * self.action_per_frame
        self.latent_channels = getattr(self.config, "latent_channels", 16)
        self.latent_f = self.frame_chunk_size
        self.latent_h = getattr(self.config, "latent_height", 16)
        self.latent_w = getattr(self.config, "latent_width", 16)
        self._h5_files: List[Path] = []
        self._index: List[Tuple[Path, str, int]] = []  # (fpath, traj_key, frame_or_start)
        self._build_index(traj_keys)

    def _collect_h5_files(self) -> List[Path]:
        """Support single file or directory (multiple H5 files)."""
        if self.h5_path.is_file() and self.h5_path.suffix.lower() in (".h5", ".hdf5"):
            return [self.h5_path]
        if self.h5_path.is_dir():
            return sorted(self.h5_path.glob("*.h5")) + sorted(self.h5_path.glob("*.hdf5"))
        return []

    def _build_index(self, traj_keys: Optional[List[str]]) -> None:
        self._h5_files = self._collect_h5_files()
        if not self._h5_files:
            raise FileNotFoundError(f"No H5 file(s) found at {self.h5_path}")

        if self.mode == "frame":
            start_f, end_f = self.frame_index_range or (0, 200)
            start_f, end_f = max(0, start_f), min(200, end_f)
            for fpath in self._h5_files:
                with h5py.File(fpath, "r") as h5:
                    keys = traj_keys if traj_keys is not None else _discover_traj_keys(h5)
                    for traj_key in keys:
                        if traj_key not in h5 or "actions" not in h5[traj_key]:
                            continue
                        n_actions = h5[traj_key]["actions"].shape[0]
                        for fi in range(start_f, min(end_f, n_actions)):
                            self._index.append((fpath, traj_key, fi))
        else:
            # mode == "chunk"
            for fpath in self._h5_files:
                with h5py.File(fpath, "r") as h5:
                    keys = traj_keys or _discover_traj_keys(h5)
                    for traj_key in keys:
                        if traj_key not in h5:
                            continue
                        g = h5[traj_key]
                        if "actions" not in g or "obs/extra/tcp_pose_wrt_base" not in g:
                            continue
                        n_actions = g["actions"].shape[0]
                        for start in range(0, n_actions - self.chunk_len + 1):
                            self._index.append((fpath, traj_key, start))

        if not self._index:
            raise ValueError(
                f"No samples found under {self.h5_path} (mode={self.mode}). "
                "For chunk mode ensure at least one traj with >= chunk_len actions and tcp_pose_wrt_base."
            )

    def __len__(self) -> int:
        return len(self._index)

    def _get_obs_extra(self, g: h5py.Group, frame_idx: int) -> Dict[str, np.ndarray]:
        out = {}
        extra = g["obs/extra"]
        for k in self.EXTRA_KEYS:
            if k in extra:
                d = extra[k]
                if d.ndim == 1:
                    out[k] = np.array(d[frame_idx], dtype=np.float32) if d.dtype != np.bool_ else np.array(d[frame_idx])
                else:
                    out[k] = np.array(d[frame_idx], dtype=np.float32)
        return out

    def _get_obs_agent(self, g: h5py.Group, frame_idx: int) -> Dict[str, np.ndarray]:
        out = {}
        agent = g["obs/agent"]
        if "qpos" in agent:
            out["qpos"] = np.array(agent["qpos"][frame_idx], dtype=np.float32)
        if self.keep_qvel and "qvel" in agent:
            out["qvel"] = np.array(agent["qvel"][frame_idx], dtype=np.float32)
        return out

    def _get_obs_images(self, g: h5py.Group, frame_idx: int) -> Dict[str, np.ndarray]:
        out = {}
        if not self.keep_images or "obs/sensor_data" not in g:
            return out
        for cam_name in ("fetch_head", "fetch_hand"):
            cam_path = f"obs/sensor_data/{cam_name}"
            if cam_path not in g:
                continue
            cam = g[cam_path]
            if "rgb" in cam:
                out[f"{cam_name}.rgb"] = np.array(cam["rgb"][frame_idx], dtype=np.uint8)
        return out

    def _get_item_frame(self, idx: int) -> Dict:
        fpath, traj_key, frame_idx = self._index[idx]
        with h5py.File(fpath, "r") as h5:
            g = h5[traj_key]
            action = np.array(g["actions"][frame_idx], dtype=np.float32)
            obs_agent = self._get_obs_agent(g, frame_idx)
            obs_extra = self._get_obs_extra(g, frame_idx)
            obs_images = self._get_obs_images(g, frame_idx)
        observation = {
            "state": obs_agent.get("qpos", np.zeros(12, dtype=np.float32)),
            "extra": obs_extra,
        }
        observation["agent"] = {"qpos": obs_agent["qpos"]}
        if obs_agent.get("qvel") is not None:
            observation["agent"]["qvel"] = obs_agent["qvel"]
        if obs_images:
            observation["images"] = obs_images
        return {
            "observation": observation,
            "action": action,
            "episode_index": _traj_key_to_episode_index(traj_key),
            "frame_index": frame_idx,
            "timestamp": float(frame_idx),
            "traj_key": traj_key,
        }

    def _get_item_chunk(self, idx: int) -> Dict:
        fpath, traj_key, start = self._index[idx]
        with h5py.File(fpath, "r") as h5:
            g = h5[traj_key]
            tcp = np.array(
                g["obs/extra/tcp_pose_wrt_base"][start + 1 : start + self.chunk_len + 1],
                dtype=np.float32,
            )
            base_action = np.array(
                g["actions"][start : start + self.chunk_len, BASE_ACTION_SLICE[0] : BASE_ACTION_SLICE[1]],
                dtype=np.float32,
            )
            trajectory_gt = np.concatenate([tcp, base_action], axis=1)
        padded = np.zeros((self.chunk_len, self.action_dim), dtype=np.float32)
        padded[:, :TRAJECTORY_DIM] = trajectory_gt
        F, N = self.frame_chunk_size, self.action_per_frame
        actions = padded.reshape(F, N, self.action_dim)
        actions = np.transpose(actions, (2, 0, 1))
        actions = actions[np.newaxis, :, :, :, np.newaxis]
        actions_mask = np.zeros((1, self.action_dim, F, N, 1), dtype=np.float32)
        actions_mask[:, :TRAJECTORY_DIM, :, :, :] = 1.0
        latents = np.zeros(
            (1, self.latent_channels, self.latent_f, self.latent_h, self.latent_w),
            dtype=np.float32,
        )
        text_emb_dim = getattr(self.config, "text_emb_dim", 4096)
        text_emb = np.zeros((1, 1, text_emb_dim), dtype=np.float32)
        return {
            "latents": torch.from_numpy(latents).float(),
            "text_emb": torch.from_numpy(text_emb).float(),
            "actions": torch.from_numpy(actions).float(),
            "actions_mask": torch.from_numpy(actions_mask).bool(),
        }

    def __getitem__(self, idx: int) -> Dict:
        if self.mode == "frame":
            return self._get_item_frame(idx)
        return self._get_item_chunk(idx)


class TrajectoryChunkDataset(MultiLeRobotTrajDataset):
    """
    Backward-compatible alias: trajectory chunk mode (latents, text_emb, actions, actions_mask).
    Use MultiLeRobotTrajDataset(h5_path, config, mode="chunk") for new code.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        config,
        *,
        frame_chunk_size: Optional[int] = None,
        action_per_frame: Optional[int] = None,
        action_dim: Optional[int] = None,
        traj_keys: Optional[List[str]] = None,
    ):
        super().__init__(
            h5_path,
            config,
            mode="chunk",
            frame_chunk_size=frame_chunk_size,
            action_per_frame=action_per_frame,
            action_dim=action_dim,
            traj_keys=traj_keys,
        )


# -----------------------------------------------------------------------------
# Optional: build index for sequence / chunk sampling (similar to lerobot_latent_dataset meta)
# -----------------------------------------------------------------------------


def build_traj_meta(
    h5_path: Union[str, Path],
    traj_keys: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Build a list of metadata entries per trajectory (for chunk-based sampling if needed).
    Each entry: {"traj_key": str, "episode_index": int, "num_frames": int, "file": Path}.
    """
    h5_path = Path(h5_path)
    files = [h5_path] if h5_path.is_file() else sorted(h5_path.glob("*.h5")) + sorted(h5_path.glob("*.hdf5"))
    meta = []
    for fpath in files:
        with h5py.File(fpath, "r") as h5:
            keys = traj_keys or _discover_traj_keys(h5)
            for traj_key in keys:
                if traj_key not in h5 or "actions" not in h5[traj_key]:
                    continue
                n = h5[traj_key]["actions"].shape[0]
                meta.append({
                    "traj_key": traj_key,
                    "episode_index": _traj_key_to_episode_index(traj_key),
                    "num_frames": n,
                    "file": fpath,
                })
    return meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str, help="Path to .h5 file or dir")
    parser.add_argument("--no-images", action="store_true", help="Do not load images")
    parser.add_argument("--no-qvel", action="store_true", help="Do not load qvel")
    args = parser.parse_args()

    ds = LeRobotTrajDataset(
        args.h5_path,
        keep_images=not args.no_images,
        keep_qvel=not args.no_qvel,
    )
    print(f"Dataset length: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())}")
            for k2, v2 in v.items():
                if isinstance(v2, np.ndarray):
                    print(f"    {k2}: shape={v2.shape}")
                elif isinstance(v2, dict):
                    print(f"    {k2}: {list(v2.keys())}")
        else:
            print(f"  {k}: {v}")
    print("observation.extra keys (required):", list(sample["observation"]["extra"].keys()))
