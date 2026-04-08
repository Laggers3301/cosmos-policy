#!/usr/bin/env python3
import argparse
import glob
import os
import random
import re

import h5py
import numpy as np
import pyarrow.parquet as pq


def _to_2d_float(column):
    values = column.to_pylist()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--dst_root", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _find_episode_mp4s(parquet_path: str):
    episode_name = os.path.splitext(os.path.basename(parquet_path))[0]
    video_base = parquet_path.replace("/data/chunk-", "/videos/chunk-")
    video_base = os.path.dirname(video_base)
    mp4s = sorted(glob.glob(os.path.join(video_base, "**", f"{episode_name}.mp4"), recursive=True))
    return mp4s


def _pick_three_cams(mp4s):
    if len(mp4s) < 3:
        return None

    lower = [m.lower() for m in mp4s]
    cam_high = None
    cam_left = None
    cam_right = None

    for p, lp in zip(mp4s, lower):
        if cam_high is None and ("high" in lp or "top" in lp or "overhead" in lp or "cam_high" in lp):
            cam_high = p
        if cam_left is None and ("left" in lp and "wrist" in lp):
            cam_left = p
        if cam_right is None and ("right" in lp and "wrist" in lp):
            cam_right = p

    if cam_high is None:
        for p, lp in zip(mp4s, lower):
            if "wrist" not in lp:
                cam_high = p
                break

    if cam_left is None or cam_right is None:
        wrist = [p for p, lp in zip(mp4s, lower) if "wrist" in lp]
        if len(wrist) >= 2:
            wrist = sorted(wrist)
            if cam_left is None:
                cam_left = wrist[0]
            if cam_right is None:
                cam_right = wrist[1]

    if cam_high and cam_left and cam_right:
        return {
            "cam_high": cam_high,
            "cam_left_wrist": cam_left,
            "cam_right_wrist": cam_right,
        }

    if len(mp4s) >= 3:
        mp4s = sorted(mp4s)
        return {
            "cam_high": mp4s[0],
            "cam_left_wrist": mp4s[1],
            "cam_right_wrist": mp4s[2],
        }
    return None


def _make_episode_hdf5(out_path, qpos, qvel, effort, action, video_paths, task_description):
    with h5py.File(out_path, "w", rdcc_nbytes=1024**2 * 2) as f:
        f.attrs["sim"] = False
        f.attrs["task_description"] = task_description
        f.attrs["success"] = True

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos.astype(np.float32))
        obs.create_dataset("qvel", data=qvel.astype(np.float32))
        obs.create_dataset("effort", data=effort.astype(np.float32))

        vp = obs.create_group("video_paths")
        for k, v in video_paths.items():
            vp.create_dataset(k, data=v.encode("utf-8"))

        f.create_dataset("action", data=action.astype(np.float32))
        rel = np.zeros_like(action, dtype=np.float32)
        if len(action) > 1:
            rel[:-1] = action[1:] - action[:-1]
            rel[-1] = rel[-2]
        f.create_dataset("relative_action", data=rel)


def main():
    args = parse_args()
    random.seed(args.seed)

    train_dir = os.path.join(args.dst_root, "train")
    val_dir = os.path.join(args.dst_root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    parquet_paths = sorted(glob.glob(os.path.join(args.src_root, "**", "data", "chunk-*", "*.parquet"), recursive=True))
    if args.limit > 0:
        parquet_paths = parquet_paths[: args.limit]

    samples = []
    skipped_no_video = 0
    skipped_bad_shape = 0
    skipped_read_error = 0
    bad_shape_examples = 0

    for i, parquet_path in enumerate(parquet_paths):
        try:
            table = pq.read_table(parquet_path)
            cols = set(table.column_names)
            needed = {
                "observation.state.arm.left.joint_positions",
                "observation.state.arm.right.joint_positions",
                "observation.state.arm.left.end_effector_value",
                "observation.state.arm.right.end_effector_value",
            }
            if not needed.issubset(cols):
                skipped_bad_shape += 1
                if bad_shape_examples < 5:
                    print(f"[skip shape] missing required cols: {parquet_path}")
                    bad_shape_examples += 1
                continue

            left_joint = _to_2d_float(table["observation.state.arm.left.joint_positions"])
            right_joint = _to_2d_float(table["observation.state.arm.right.joint_positions"])
            left_gripper = _to_2d_float(table["observation.state.arm.left.end_effector_value"])
            right_gripper = _to_2d_float(table["observation.state.arm.right.end_effector_value"])

            if left_joint.ndim != 2 or right_joint.ndim != 2:
                skipped_bad_shape += 1
                if bad_shape_examples < 5:
                    print(
                        f"[skip shape] joint ndim invalid left={left_joint.ndim} right={right_joint.ndim}: {parquet_path}"
                    )
                    bad_shape_examples += 1
                continue

            T = left_joint.shape[0]
            if (
                right_joint.shape[0] != T
                or left_gripper.shape[0] != T
                or right_gripper.shape[0] != T
            ):
                skipped_bad_shape += 1
                if bad_shape_examples < 5:
                    print(
                        f"[skip shape] length mismatch left={left_joint.shape[0]} right={right_joint.shape[0]} "
                        f"lg={left_gripper.shape[0]} rg={right_gripper.shape[0]}: {parquet_path}"
                    )
                    bad_shape_examples += 1
                continue

            if left_joint.shape[1] == 6:
                left = np.concatenate([left_joint, left_gripper], axis=1)
            elif left_joint.shape[1] == 7:
                left = left_joint
            else:
                skipped_bad_shape += 1
                if bad_shape_examples < 5:
                    print(f"[skip shape] left joint dim={left_joint.shape[1]}: {parquet_path}")
                    bad_shape_examples += 1
                continue

            if right_joint.shape[1] == 6:
                right = np.concatenate([right_joint, right_gripper], axis=1)
            elif right_joint.shape[1] == 7:
                right = right_joint
            else:
                skipped_bad_shape += 1
                if bad_shape_examples < 5:
                    print(f"[skip shape] right joint dim={right_joint.shape[1]}: {parquet_path}")
                    bad_shape_examples += 1
                continue

            qpos = np.concatenate([left, right], axis=1)
            qvel = np.zeros_like(qpos, dtype=np.float32)
            effort = np.zeros_like(qpos, dtype=np.float32)
            action = np.zeros_like(qpos, dtype=np.float32)
            if len(qpos) > 1:
                action[:-1] = qpos[1:]
                action[-1] = qpos[-1]
            else:
                action[:] = qpos

            mp4s = _find_episode_mp4s(parquet_path)
            cams = _pick_three_cams(mp4s)
            if cams is None:
                skipped_no_video += 1
                if skipped_no_video <= 5:
                    print(f"[skip video] {parquet_path}")
                continue

            task_desc = "fold shirt"
            samples.append((qpos, qvel, effort, action, cams, task_desc))

            if (i + 1) % 500 == 0:
                print(f"processed {i+1}/{len(parquet_paths)}; usable={len(samples)}")
        except Exception:
            skipped_read_error += 1
            continue

    print(f"total parquet: {len(parquet_paths)}")
    print(f"usable episodes: {len(samples)}")
    print(f"skipped_no_video: {skipped_no_video}")
    print(f"skipped_bad_shape: {skipped_bad_shape}")
    print(f"skipped_read_error: {skipped_read_error}")

    if not samples:
        print("No usable episodes found. Please inspect video folder naming under src_root.")
        return

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_indices = set(range(n_val))

    tr = 0
    va = 0
    for idx, sample in enumerate(samples):
        qpos, qvel, effort, action, cams, task_desc = sample
        if idx in val_indices:
            out_path = os.path.join(val_dir, f"episode_{va}.hdf5")
            va += 1
        else:
            out_path = os.path.join(train_dir, f"episode_{tr}.hdf5")
            tr += 1
        _make_episode_hdf5(out_path, qpos, qvel, effort, action, cams, task_desc)

    print(f"done. train={tr}, val={va}")


if __name__ == "__main__":
    main()
