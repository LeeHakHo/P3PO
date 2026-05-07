import h5py
import cv2
import numpy as np
import pickle
from pathlib import Path
from PIL import Image

DATA_DIR = "/scr/hyeonhoo/code/P3PO/data/2026-05-01"
OUTPUT_DIR = "/scr/hyeonhoo/code/P3PO/data/general"
TASK_NAME = "real_robot"
CAM_MAIN = "37998989"
CAM_WRIST = "14064085"
IMG_SIZE = 128


def read_mp4_frames(mp4_path, img_size):
    frames = []
    cap = cv2.VideoCapture(str(mp4_path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)


def read_depth_frames(depth_dir, n_steps, img_size):
    depths = []
    for i in range(n_steps):
        depth_path = depth_dir / f"{i}.png"
        if depth_path.exists():
            depth = np.array(Image.open(depth_path))
            depth = cv2.resize(depth, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        else:
            depth = np.zeros((img_size, img_size), dtype=np.uint16)
        depths.append(depth)
    return np.array(depths)


data_dir = Path(DATA_DIR)
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

all_actions = []
all_observations = []

episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
print(f"총 {len(episode_dirs)}개 에피소드 발견")

for ep_idx, ep_dir in enumerate(episode_dirs):
    h5_path = ep_dir / "trajectory.h5"
    mp4_main = ep_dir / "recordings" / "MP4" / f"{CAM_MAIN}.mp4"
    mp4_wrist = ep_dir / "recordings" / "MP4" / f"{CAM_WRIST}.mp4"
    depth_main_dir = ep_dir / "recordings" / "Depth" / CAM_MAIN

    if not h5_path.exists() or not mp4_main.exists():
        print(f"  스킵: {ep_dir.name} (파일 없음)")
        continue

    with h5py.File(h5_path, "r") as f:
        cartesian = f["action/cartesian_position"][:]  # (N, 6)
        gripper = f["action/gripper_position"][:]      # (N,)

    action = np.concatenate([cartesian, gripper[:, None]], axis=1)  # (N, 7)
    n_steps = len(action)

    pixels_main = read_mp4_frames(mp4_main, IMG_SIZE)
    pixels_wrist = read_mp4_frames(mp4_wrist, IMG_SIZE) if mp4_wrist.exists() else None

    # 길이 맞추기
    min_len = min(n_steps, len(pixels_main))
    if pixels_wrist is not None:
        min_len = min(min_len, len(pixels_wrist))

    action = action[:min_len]
    pixels_main = pixels_main[:min_len]
    if pixels_wrist is not None:
        pixels_wrist = pixels_wrist[:min_len]

    if depth_main_dir.exists():
        depth = read_depth_frames(depth_main_dir, min_len, IMG_SIZE)
    else:
        depth = np.zeros((min_len, IMG_SIZE, IMG_SIZE), dtype=np.uint16)

    obs = {
        "pixels": pixels_main,
        "pixels_wrist": pixels_wrist if pixels_wrist is not None else np.zeros_like(pixels_main),
        "depth": depth,
    }

    all_actions.append(action)
    all_observations.append(obs)
    print(f"  [{ep_idx+1}/{len(episode_dirs)}] {ep_dir.name}: {min_len} 스텝")

data = {
    "actions": all_actions,
    "observations": all_observations,
}

output_path = output_dir / f"{TASK_NAME}.pkl"
with open(output_path, "wb") as f:
    pickle.dump(data, f)
print(f"\n저장 완료: {output_path}")
print(f"총 {len(all_actions)}개 에피소드")
