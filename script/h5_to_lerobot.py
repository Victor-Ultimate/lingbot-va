#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Convert H5 trajectory dataset to LeRobot format.

Input H5 structure (per traj_XXX):
  obs/agent: qpos, qvel
  obs/extra: tcp_pose_wrt_base, obj_pose_wrt_base, goal_pos_wrt_base, is_grasped
  obs/sensor_data/fetch_head/rgb, fetch_hand/rgb  (optional)
  actions: (T, 13)

Output LeRobot layout:
  your_dataset/
  ├── videos/
  │   └── chunk-XXX/
  │       └── observation.images.<cam_name>/
  │           └── episode_YYYYYY.mp4
  └── meta/
      └── episodes.jsonl

Usage:
  python script/h5_to_lerobot.py /path/to/h5_or_dir /path/to/output_dir [--chunk-size 100] [--fps 30]
"""

from pathlib import Path
import re
import json
import argparse
import h5py
import numpy as np

# Camera name in H5 -> folder name under observation.images (LeRobot style)
CAM_MAPPING = {
    "fetch_head": "cam_high",
    "fetch_hand": "cam_low",
}


def discover_traj_keys(h5: h5py.File):
    keys = [k for k in h5.keys() if isinstance(h5[k], h5py.Group) and re.match(r"traj_\d+", k)]
    return sorted(keys, key=lambda x: int(x.split("_")[1]))


def collect_h5_files(h5_path: Path):
    if h5_path.is_file() and h5_path.suffix.lower() in (".h5", ".hdf5"):
        return [h5_path]
    if h5_path.is_dir():
        return sorted(h5_path.glob("*.h5")) + sorted(h5_path.glob("*.hdf5"))
    return []


def get_episode_list(h5_path: Path):
    """Yield (fpath, traj_key) for each episode across all H5 files."""
    files = collect_h5_files(h5_path)
    if not files:
        raise FileNotFoundError(f"No H5 file(s) found at {h5_path}")
    for fpath in files:
        with h5py.File(fpath, "r") as h5:
            for traj_key in discover_traj_keys(h5):
                if traj_key not in h5 or "actions" not in h5[traj_key]:
                    continue
                yield fpath, traj_key


def get_available_cameras(g: h5py.Group):
    out = []
    if "obs/sensor_data" not in g:
        return out
    for h5_cam in ("fetch_head", "fetch_hand"):
        cam_path = f"obs/sensor_data/{h5_cam}"
        if cam_path in g and "rgb" in g[cam_path]:
            out.append((h5_cam, CAM_MAPPING.get(h5_cam, h5_cam)))
    return out


def write_video_mp4(frames: np.ndarray, out_path: Path, fps: int = 30):
    """Write (N, H, W, 3) uint8 to mp4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
    except ImportError:
        try:
            import cv2
        except ImportError:
            raise ImportError("Need imageio or opencv-python to write mp4. pip install imageio[ffmpeg]")
        # cv2 path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frames.shape[1], frames.shape[2]
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for i in range(frames.shape[0]):
            writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        writer.release()
        return
    # imageio path (RGB)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", format="FFMPEG")
    for i in range(frames.shape[0]):
        writer.append_data(frames[i])
    writer.close()


def convert(h5_path: Path, out_dir: Path, chunk_size: int = 100, fps: int = 30, task_text: str = "trajectory"):
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    videos_dir = out_dir / "videos"
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    global_episode_index = 0
    chunk_index = 0
    episodes_in_chunk = 0

    for fpath, traj_key in get_episode_list(h5_path):
        with h5py.File(fpath, "r") as h5:
            g = h5[traj_key]
            n_actions = g["actions"].shape[0]
            cameras = get_available_cameras(g)

            # Episode length = number of frames we export (match action steps; video can be n_actions or n_actions+1)
            n_frames = n_actions
            if cameras:
                rgb0 = g[f"obs/sensor_data/{cameras[0][0]}/rgb"]
                n_frames = min(rgb0.shape[0], n_actions + 1)

            # Chunk folder: chunk-000, chunk-001, ...
            if episodes_in_chunk >= chunk_size:
                chunk_index += 1
                episodes_in_chunk = 0
            chunk_name = f"chunk-{chunk_index:03d}"
            episode_name = f"episode_{global_episode_index:06d}"

            for h5_cam, lerobot_cam in cameras:
                rgb = np.array(g[f"obs/sensor_data/{h5_cam}/rgb"], dtype=np.uint8)
                if rgb.ndim == 3:
                    rgb = np.stack([rgb] * 3, axis=-1)  # (T,H,W) -> (T,H,W,3)
                n_export = min(rgb.shape[0], n_frames)
                video_dir = videos_dir / chunk_name / f"observation.images.{lerobot_cam}"
                video_path = video_dir / f"{episode_name}.mp4"
                write_video_mp4(rgb[:n_export], video_path, fps=fps)

            # meta/episodes.jsonl line
            length = n_frames
            action_config = [
                {"start_frame": 0, "end_frame": length, "action_text": task_text}
            ]
            episodes_meta.append({
                "episode_index": global_episode_index,
                "tasks": [task_text],
                "length": length,
                "action_config": action_config,
            })

            global_episode_index += 1
            episodes_in_chunk += 1

    # Write meta/episodes.jsonl
    meta_file = meta_dir / "episodes.jsonl"
    with open(meta_file, "w") as f:
        for rec in episodes_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Converted {global_episode_index} episodes to {out_dir}")
    print(f"  videos: {videos_dir}")
    print(f"  meta:   {meta_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert H5 trajectory dataset to LeRobot format")
    parser.add_argument("h5_path", type=str, help="Path to .h5 file or directory of .h5 files")
    parser.add_argument("out_dir", type=str, help="Output LeRobot dataset root (videos/ + meta/)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Episodes per chunk folder (default 100)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default 30)")
    parser.add_argument("--task-text", type=str, default="trajectory", help="Default action_text for action_config")
    args = parser.parse_args()
    convert(Path(args.h5_path), Path(args.out_dir), chunk_size=args.chunk_size, fps=args.fps, task_text=args.task_text)


if __name__ == "__main__":
    main()
