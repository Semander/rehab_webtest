"""Stage 1: batch MediaPipe pose extraction for REHAB24-6 reps.

Driven by the rep manifest built during data exploration. For each annotated
rep, runs MediaPipe Pose (VIDEO mode) on the frames inside the rep window and
caches per-frame landmarks to disk so downstream stages never re-run inference.

Camera selection follows the exploration report:
- front-orientation sessions: Camera18 only (sagittal view)
- half-profile sessions: both cameras; the one with higher mean landmark
  visibility across the rep becomes the primary

Output: one compressed .npz per (rep, camera) with
- image_landmarks: (n_frames, 33, 4) normalized x, y, z, visibility
- world_landmarks: (n_frames, 33, 4) metric x, y, z, visibility
- frame_indices, plus video width/height/fps metadata
Frames where detection fails are NaN.

An updated manifest (ex6_rep_manifest_pose.csv) records the chosen camera,
pose file paths, mean knee visibility, and a pose_ok quality flag.

Usage (from repo root):
    .venv/Scripts/python -m rehab_ai.pose_extraction --model full
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np

from config.settings import ROOT, ModelTypePath
from rehab_ai.pose_estimator import PoseEstimator

N_LANDMARKS = 33
LEFT_KNEE, RIGHT_KNEE = 25, 26
KNEE_VISIBILITY_THRESHOLD = 0.6

DATASET_ROOT = ROOT / "data" / "SYDE660A_Dataset"
MANIFEST_PATH = DATASET_ROOT / "exploration" / "manifests" / "ex6_rep_manifest.csv"
OUTPUT_DIR = ROOT / "data" / "poses" / "ex6"
OUTPUT_MANIFEST = DATASET_ROOT / "exploration" / "manifests" / "ex6_rep_manifest_pose.csv"

MODEL_PATHS = {
    "lite": ModelTypePath.LITE,
    "full": ModelTypePath.FULL,
    "heavy": ModelTypePath.HEAVY,
}


def landmarks_to_array(landmark_list):
    arr = np.full((N_LANDMARKS, 4), np.nan, dtype=np.float32)
    for i, lm in enumerate(landmark_list):
        arr[i] = (lm.x, lm.y, lm.z, lm.visibility)
    return arr


def cameras_for_rep(rep):
    if rep["sagittal_view_camera"] == "Camera18":
        return ["cam18"]
    return ["cam17", "cam18"]  # half-profile: extract both, pick later


def extract_video(video_path, rep_windows, model_path):
    """Run pose detection on one video file over the union of rep windows.

    rep_windows: list of (first_frame, last_frame) inclusive.
    Returns dict frame_idx -> (image_landmarks, world_landmarks), video meta,
    and measured detection throughput in frames per second.
    """
    wanted = set()
    for first, last in rep_windows:
        wanted.update(range(first, last + 1))
    max_frame = max(wanted)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    meta = {
        "width": int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
    }

    estimator = PoseEstimator(model_path, running_mode=mp.tasks.vision.RunningMode.VIDEO)
    results = {}
    detect_time = 0.0
    frame_idx = 0
    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            t0 = time.perf_counter()
            pose = estimator.detect_pose(mp_image, timestamp_ms)
            detect_time += time.perf_counter() - t0
            if pose.landmarks_list:
                results[frame_idx] = (
                    landmarks_to_array(pose.landmarks_list[0]),
                    landmarks_to_array(pose.world_landmarks_list[0]),
                )
        frame_idx += 1
    cap.release()
    estimator.close()

    detected = len(results)
    throughput = detected / detect_time if detect_time > 0 else 0.0
    return results, meta, throughput


def rep_output_path(out_dir, video_id, rep_num, camera):
    return out_dir / f"{video_id}_rep{int(rep_num):02d}_{camera}.npz"


def rep_stats_from_file(out_path):
    """Recompute rep stats from an already-saved npz (for resumed runs)."""
    d = np.load(out_path)
    return _rep_stats(out_path, d["image_landmarks"])


def save_rep(out_dir, video_id, rep_num, camera, first, last, frame_results, meta):
    """Slice one rep out of per-video results and save it. Returns stats."""
    n = last - first + 1
    image_lm = np.full((n, N_LANDMARKS, 4), np.nan, dtype=np.float32)
    world_lm = np.full((n, N_LANDMARKS, 4), np.nan, dtype=np.float32)
    for i, f in enumerate(range(first, last + 1)):
        if f in frame_results:
            image_lm[i], world_lm[i] = frame_results[f]

    out_path = rep_output_path(out_dir, video_id, rep_num, camera)
    np.savez_compressed(
        out_path,
        image_landmarks=image_lm,
        world_landmarks=world_lm,
        frame_indices=np.arange(first, last + 1),
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
    )
    return _rep_stats(out_path, image_lm)


def _rep_stats(out_path, image_lm):
    try:
        rel_path = out_path.relative_to(ROOT).as_posix()
    except ValueError:  # output dir outside the repo
        rel_path = out_path.as_posix()

    vis = image_lm[:, :, 3]
    knee_vis = np.nanmean(vis[:, [LEFT_KNEE, RIGHT_KNEE]], axis=0)
    detected_frames = int(np.sum(~np.isnan(vis[:, 0])))
    return {
        "path": rel_path,
        "mean_visibility": float(np.nanmean(vis)) if detected_frames else 0.0,
        "left_knee_visibility": float(knee_vis[0]) if detected_frames else 0.0,
        "right_knee_visibility": float(knee_vis[1]) if detected_frames else 0.0,
        "detected_frames": detected_frames,
        "n_frames": image_lm.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_PATHS, default="full")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, newline="") as f:
        reps = list(csv.DictReader(f))

    # group work by physical video file: (video_id, camera) -> reps
    tasks = defaultdict(list)
    for rep in reps:
        for camera in cameras_for_rep(rep):
            tasks[(rep["video_id"], camera)].append(rep)

    rep_stats = {}  # (video_id, rep_num, camera) -> stats
    throughputs = []
    for (video_id, camera), video_reps in sorted(tasks.items()):
        existing = [rep_output_path(args.out, video_id, r["repetition_number"], camera)
                    for r in video_reps]
        if all(p.exists() for p in existing):
            print(f"{video_id} {camera}: {len(video_reps)} reps already extracted, "
                  f"loading stats from cache", flush=True)
            for rep, p in zip(video_reps, existing):
                rep_stats[(video_id, rep["repetition_number"], camera)] = rep_stats_from_file(p)
            continue

        path_col = "video_path_cam17" if camera == "cam17" else "video_path_cam18"
        video_path = DATASET_ROOT / video_reps[0][path_col]
        windows = [(int(r["first_frame"]), int(r["last_frame"])) for r in video_reps]
        n_frames = sum(last - first + 1 for first, last in windows)
        print(f"{video_id} {camera}: {len(video_reps)} reps, {n_frames} frames "
              f"from {video_path.name}", flush=True)

        frame_results, meta, throughput = extract_video(video_path, windows, model_path)
        throughputs.append(throughput)
        print(f"  detected {len(frame_results)}/{n_frames} frames "
              f"at {throughput:.1f} fps", flush=True)

        for rep in video_reps:
            first, last = int(rep["first_frame"]), int(rep["last_frame"])
            stats = save_rep(args.out, video_id, rep["repetition_number"],
                             camera, first, last, frame_results, meta)
            rep_stats[(video_id, rep["repetition_number"], camera)] = stats

    # build updated manifest: pick primary camera, apply quality gate
    out_rows = []
    for rep in reps:
        cameras = cameras_for_rep(rep)
        stats_by_cam = {c: rep_stats[(rep["video_id"], rep["repetition_number"], c)]
                        for c in cameras}
        primary = max(cameras, key=lambda c: stats_by_cam[c]["mean_visibility"])
        s = stats_by_cam[primary]
        row = dict(rep)
        row.update({
            "pose_camera": primary,
            "pose_path": s["path"],
            "pose_path_secondary": (stats_by_cam[[c for c in cameras if c != primary][0]]["path"]
                                    if len(cameras) > 1 else ""),
            "pose_mean_visibility": round(s["mean_visibility"], 4),
            "pose_left_knee_visibility": round(s["left_knee_visibility"], 4),
            "pose_right_knee_visibility": round(s["right_knee_visibility"], 4),
            "pose_detected_frames": s["detected_frames"],
            # unreliable only when BOTH knees are poorly visible; in a sagittal
            # view the far knee is expected to be occluded
            "pose_ok": max(s["left_knee_visibility"],
                           s["right_knee_visibility"]) >= KNEE_VISIBILITY_THRESHOLD,
        })
        out_rows.append(row)

    with open(OUTPUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    n_ok = sum(r["pose_ok"] for r in out_rows)
    print(f"\ndone: {len(out_rows)} reps, pose_ok {n_ok}/{len(out_rows)}")
    if throughputs:
        print(f"mean detection throughput: {np.mean(throughputs):.1f} fps ({args.model} model)")
    print(f"manifest written to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
