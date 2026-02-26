# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Config for trajectory training: GT = end-effector trajectory (tcp_pose_wrt_base + base_action)."""
from easydict import EasyDict
from .va_robotwin_train_cfg import va_robotwin_train_cfg
import os

va_robotwin_trajectory_train_cfg = EasyDict(__name__='Config: VA robotwin trajectory train')
va_robotwin_trajectory_train_cfg.update(va_robotwin_train_cfg)

# H5 trajectory dataset
va_robotwin_trajectory_train_cfg.h5_dataset_path = '/mnt/public/data/h200/victor/mshab/replica/scene_datasets/replica_cad_dataset/rearrange-dataset/prepare_groceries/pick/002_master_chef_can.h5'  # .h5 file or dir
va_robotwin_trajectory_train_cfg.trajectory_dim = 9  # tcp_pose_wrt_base (7) + base_action (2)
va_robotwin_trajectory_train_cfg.latent_loss_weight = 0.0  # trajectory-only: no video loss
va_robotwin_trajectory_train_cfg.latent_channels = 16
va_robotwin_trajectory_train_cfg.latent_height = 16
va_robotwin_trajectory_train_cfg.latent_width = 16
va_robotwin_trajectory_train_cfg.text_emb_dim = 4096
