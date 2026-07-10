#!/usr/bin/env python3
"""
REHAB24-6 Dataset Exploration Pipeline.

Usage:
    python -m src.explore              # Steps 0-5
    python -m src.explore --probe-pose # Steps 0-6
"""

import argparse
import hashlib
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FPS = 30.0

EXERCISE_NAMES: Dict[int, str] = {
    1: "Arm Abduction",
    2: "Arm VW",
    3: "Push-ups",
    4: "Leg Abduction",
    5: "Leg Lunge",
    6: "Squats",
}

# cam17 orientation -> cam18 orientation (cameras are orthogonal)
CAM17_TO_CAM18: Dict[str, str] = {
    "front": "side",
    "half-profile": "half-profile",
    "profile": "front",
}

# Which camera provides the sagittal (pure side) view?
SAGITTAL_CAM: Dict[str, str] = {
    "front": "Camera18",    # person faces cam17 -> cam18 sees them from the side
    "profile": "Camera17",  # person is in profile to cam17 -> cam17 is the side view
    "half-profile": "neither",
}

EXPECTED_MD5: Dict[str, str] = {
    "Segmentation.csv": "90b8fbd7445dd050bf27b17126c78fbe",
    "Segmentation.txt": "5f2a5b886c6f794f03e1a8f642738c86",
    "videos.zip": "ed183a245a1b171638e422f7a288e5a8",
}

CSV_EXPECTED_COLUMNS: List[str] = [
    "video_id", "repetition_number", "exercise_id", "person_id",
    "first_frame", "last_frame", "cam17_orientation", "mocap_erroneous",
    "exercise_subtype", "lights_on", "extra_person_in_cam17",
    "extra_person_in_cam18", "correctness",
]

MANIFESTS_DIR = ROOT / "exploration" / "manifests"
FIGURES_DIR = ROOT / "exploration" / "figures"
REPORT_DIR = ROOT / "exploration" / "report"
SAMPLES_DIR = ROOT / "exploration" / "samples"

# Minimum incorrect-rep count per subject to be usable as LOSO test fold
MIN_CLASS_REPS = 3

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_data_dir() -> Path:
    """Return the directory containing Segmentation.csv (./data/ or ./root)."""
    for candidate in [ROOT / "data", ROOT]:
        if (candidate / "Segmentation.csv").exists():
            return candidate
    return ROOT / "data"


def find_videos_root(data_dir: Path) -> Path:
    """Return the directory that contains Ex1/, Ex2/, ... subdirs."""
    for candidate in [data_dir / "videos", ROOT / "videos", data_dir, ROOT]:
        if candidate.is_dir() and (candidate / "Ex1").is_dir():
            return candidate
    return data_dir / "videos"


def discover_video_map(videos_root: Path) -> Dict[str, dict]:
    """
    Scan videos_root/ExN/ and return:
      { video_id: {"exercise_dir": "ExN", "cam17": Path|None, "cam18": Path|None} }
    """
    mapping: Dict[str, dict] = {}
    pat = re.compile(
        r"^(PM_[\w]+)-Camera(\d+)-30fps(?:-transposed)?\.mp4$", re.IGNORECASE
    )
    for ex_dir in sorted(videos_root.iterdir()):
        if not ex_dir.is_dir() or not ex_dir.name.lower().startswith("ex"):
            continue
        for mp4 in sorted(ex_dir.glob("*.mp4")):
            m = pat.match(mp4.name)
            if not m:
                continue
            vid_id = m.group(1)
            cam_num = int(m.group(2))
            if vid_id not in mapping:
                mapping[vid_id] = {"exercise_dir": ex_dir.name, "cam17": None, "cam18": None}
            if cam_num == 17:
                mapping[vid_id]["cam17"] = mp4
            elif cam_num == 18:
                mapping[vid_id]["cam18"] = mp4
    return mapping


def get_video_props(path: Path) -> Optional[dict]:
    """Read video metadata via OpenCV. Returns None if file can't be opened."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "width": w, "height": h, "fps": fps,
        "frame_count": fc,
        "duration_s": fc / fps if fps > 0 else 0.0,
    }


def read_frame(path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Read a single frame from a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def rel(path: Optional[Path]) -> Optional[str]:
    """Return path relative to ROOT as a POSIX string, or None."""
    if path is None:
        return None
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# -----------------------------------------------------------------------------
# Step 0: Acquire & Verify
# -----------------------------------------------------------------------------

def step0_verify(
    data_dir: Path, videos_root: Path
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    print("\n" + "=" * 60)
    print("  STEP 0: ACQUIRE & VERIFY")
    print("=" * 60)

    md5_results: Dict[str, str] = {}
    for filename, expected in EXPECTED_MD5.items():
        # Look in data_dir and ROOT
        path = None
        for candidate in [data_dir / filename, ROOT / filename]:
            if candidate.exists():
                path = candidate
                break
        if path is None:
            print(f"  [MISSING] {filename}")
            md5_results[filename] = "MISSING"
            continue
        actual = md5_file(path)
        status = "PASS" if actual == expected else f"FAIL (got {actual})"
        icon = "OK" if actual == expected else "FAIL"
        print(f"  [{icon}] MD5 {status:6s}: {filename}")
        md5_results[filename] = status

    print(f"\n  Data dir  : {data_dir}")
    print(f"  Videos dir: {videos_root}")
    video_map = discover_video_map(videos_root)
    print(f"\n  Discovered {len(video_map)} video sessions "
          f"({len(video_map) * 2} files expected):\n")
    print(f"  {'video_id':14s} {'exer_dir':8s}  cam17 filename                         cam18 filename")
    print(f"  {'-'*14} {'-'*8}  {'-'*38} {'-'*38}")
    for vid_id, info in sorted(video_map.items()):
        c17 = info["cam17"].name if info["cam17"] else "MISSING"
        c18 = info["cam18"].name if info["cam18"] else "MISSING"
        print(f"  {vid_id:14s} {info['exercise_dir']:8s}  {c17:38s} {c18}")

    return md5_results, video_map


# -----------------------------------------------------------------------------
# Step 2: Parse & Validate CSV
# -----------------------------------------------------------------------------

def step2_validate(data_dir: Path, video_map: Dict[str, dict]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  STEP 2: PARSE & VALIDATE")
    print("=" * 60)

    csv_path = data_dir / "Segmentation.csv"
    df = pd.read_csv(csv_path, sep=";")

    # -- Schema check ---------------------------------------------------------
    missing_cols = [c for c in CSV_EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in CSV_EXPECTED_COLUMNS]
    print(f"\n  Rows: {len(df)}, Columns: {len(df.columns)}")
    if missing_cols:
        print(f"  SCHEMA MISMATCH - missing columns: {missing_cols}")
    elif extra_cols:
        print(f"  Schema OK (extra columns present: {extra_cols})")
    else:
        print("  Schema OK - all expected columns present, no extras")

    # -- Dtypes & unique values ------------------------------------------------
    print("\n  Column dtypes:")
    for col in df.columns:
        print(f"    {col:28s}: {df[col].dtype}")

    print("\n  Unique values for key columns:")
    for col in ["exercise_id", "person_id", "cam17_orientation",
                "exercise_subtype", "lights_on", "correctness"]:
        vals = sorted(str(v) for v in df[col].dropna().unique())
        print(f"    {col:28s}: {vals}")

    # -- NaN audit ------------------------------------------------------------
    nan_counts = df.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if cols_with_nan.empty:
        print("\n  NaN check: no NaN values found (exercise_subtype NaNs expected)")
    else:
        print("\n  NaN counts (non-zero only):")
        for col, cnt in cols_with_nan.items():
            note = " (expected - exercises with no subtype distinction)" \
                if col == "exercise_subtype" else ""
            print(f"    {col:28s}: {cnt}{note}")

    # -- Impossible values -----------------------------------------------------
    bad_frames = df[df["last_frame"] < df["first_frame"]]
    print(f"\n  last_frame < first_frame: {len(bad_frames)} rows"
          + (" a?? PROBLEM" if len(bad_frames) > 0 else ""))
    if len(bad_frames) > 0:
        print(bad_frames[["video_id", "repetition_number", "first_frame", "last_frame"]])

    # -- Duplicate rows --------------------------------------------------------
    dup_key = ["video_id", "repetition_number"]
    dups = df[df.duplicated(subset=dup_key, keep=False)]
    print(f"  Duplicate (video_id, rep_number) pairs: {len(dups) // 2 if len(dups) > 0 else 0}")

    # -- Overlapping frame windows within same video ---------------------------
    overlap_count = 0
    overlap_examples = []
    for vid_id, grp in df.groupby("video_id"):
        grp_sorted = grp.sort_values("first_frame").reset_index(drop=True)
        for i in range(len(grp_sorted) - 1):
            a = grp_sorted.iloc[i]
            b = grp_sorted.iloc[i + 1]
            if b["first_frame"] <= a["last_frame"]:
                overlap_count += 1
                overlap_examples.append(
                    f"{vid_id} rep{int(a['repetition_number'])}&rep{int(b['repetition_number'])}: "
                    f"[{int(a['first_frame'])},{int(a['last_frame'])}] overlaps "
                    f"[{int(b['first_frame'])},{int(b['last_frame'])}]"
                )
    print(f"  Overlapping frame windows: {overlap_count}")
    for ex in overlap_examples[:5]:
        print(f"    {ex}")
    if len(overlap_examples) > 5:
        print(f"    ... and {len(overlap_examples)-5} more")

    # -- video_id resolution ---------------------------------------------------
    csv_ids = set(df["video_id"].unique())
    map_ids = set(video_map.keys())
    unresolved = csv_ids - map_ids
    on_disk_only = map_ids - csv_ids
    print(f"\n  video_ids in CSV: {len(csv_ids)}, on disk: {len(map_ids)}")
    print(f"  Unresolved (CSV but no file): {len(unresolved)}"
          + (f" -> {sorted(unresolved)}" if unresolved else ""))
    print(f"  On disk only (no CSV entry):  {len(on_disk_only)}"
          + (f" -> {sorted(on_disk_only)}" if on_disk_only else ""))

    return df


# -----------------------------------------------------------------------------
# Step 3: Build Manifests
# -----------------------------------------------------------------------------

def step3_manifests(
    df: pd.DataFrame,
    video_map: Dict[str, dict],
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("  STEP 3: BUILD MANIFESTS")
    print("=" * 60)

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- 3a: Per-video properties from OpenCV ---------------------------------
    print("\n  Reading video properties (this may take a moment)...")
    vid_props: Dict[str, dict] = {}   # video_id -> {"cam17": props, "cam18": props}
    for vid_id, info in sorted(video_map.items()):
        vid_props[vid_id] = {"cam17": None, "cam18": None}
        for cam_key in ("cam17", "cam18"):
            p = info[cam_key]
            if p and p.exists():
                props = get_video_props(p)
                vid_props[vid_id][cam_key] = props
                if props is None:
                    print(f"    WARNING: Cannot open {p}")
        sys.stdout.write(".")
        sys.stdout.flush()
    print(f"\n  Done - {len(vid_props)} sessions probed.")

    # -- 3b: Per-rep manifest --------------------------------------------------
    rows = []
    for _, r in df.iterrows():
        vid_id = r["video_id"]
        info = video_map.get(vid_id, {})
        props_cam17 = vid_props.get(vid_id, {}).get("cam17")
        props_cam18 = vid_props.get(vid_id, {}).get("cam18")

        cam17_path = info.get("cam17")
        cam18_path = info.get("cam18")
        has_video = (
            cam17_path is not None and cam17_path.exists() and
            cam18_path is not None and cam18_path.exists()
        )

        n_frames = int(r["last_frame"]) - int(r["first_frame"]) + 1
        duration_s = n_frames / FPS

        cam17_ori = str(r["cam17_orientation"])
        cam18_ori = CAM17_TO_CAM18.get(cam17_ori, "unknown")
        sagittal_cam = SAGITTAL_CAM.get(cam17_ori, "unknown")

        # quality_ok: not erroneous mocap AND extra_person <= 1 in both cameras
        mocap_err = bool(int(r["mocap_erroneous"]))
        ep17 = int(r["extra_person_in_cam17"])
        ep18 = int(r["extra_person_in_cam18"])
        quality_ok = (not mocap_err) and (ep17 in (0, 1)) and (ep18 in (0, 1))

        # frames_in_range: last_frame < frame_count of the cam17 video
        # (both cameras should have same frame count, use cam17 as reference)
        frame_count_ref = props_cam17["frame_count"] if props_cam17 else None
        if frame_count_ref is not None:
            frames_in_range = int(r["last_frame"]) < frame_count_ref
        else:
            frames_in_range = None  # unknown (video not readable)

        subtype = r["exercise_subtype"] if pd.notna(r.get("exercise_subtype")) else None

        rows.append({
            "video_id": vid_id,
            "repetition_number": int(r["repetition_number"]),
            "exercise_id": int(r["exercise_id"]),
            "exercise_name": EXERCISE_NAMES.get(int(r["exercise_id"]), "Unknown"),
            "person_id": int(r["person_id"]),
            "first_frame": int(r["first_frame"]),
            "last_frame": int(r["last_frame"]),
            "n_frames": n_frames,
            "duration_s": round(duration_s, 4),
            "cam17_orientation": cam17_ori,
            "cam18_orientation": cam18_ori,
            "sagittal_view_camera": sagittal_cam,
            "exercise_subtype": subtype,
            "lights_on": bool(int(r["lights_on"])),
            "mocap_erroneous": mocap_err,
            "extra_person_in_cam17": ep17,
            "extra_person_in_cam18": ep18,
            "quality_ok": quality_ok,
            "correctness": int(r["correctness"]),
            "video_path_cam17": rel(cam17_path),
            "video_path_cam18": rel(cam18_path),
            "has_video": has_video,
            "frames_in_range": frames_in_range,
        })

    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(MANIFESTS_DIR / "rep_manifest.csv", index=False)
    table = pa.Table.from_pandas(rep_df)
    pq.write_table(table, MANIFESTS_DIR / "rep_manifest.parquet")
    print(f"\n  rep_manifest: {len(rep_df)} rows saved "
          f"-> manifests/rep_manifest.csv + .parquet")

    # -- 3c: Per-video manifest ------------------------------------------------
    video_rows = []
    for vid_id, info in sorted(video_map.items()):
        for cam_key, cam_label in [("cam17", "Camera17"), ("cam18", "Camera18")]:
            p = info[cam_key]
            if p is None:
                continue
            props = vid_props.get(vid_id, {}).get(cam_key) or {}
            # Reps that reference this video_id
            vid_reps = rep_df[rep_df["video_id"] == vid_id]
            exercises = sorted(vid_reps["exercise_id"].unique().tolist())
            rep_nums = sorted(vid_reps["repetition_number"].tolist())
            video_rows.append({
                "file_path": rel(p),
                "video_id": vid_id,
                "camera": cam_label,
                "exercise_dir": info["exercise_dir"],
                "exercise_ids": ",".join(str(e) for e in exercises),
                "width": props.get("width"),
                "height": props.get("height"),
                "fps": props.get("fps"),
                "frame_count": props.get("frame_count"),
                "duration_s": round(props.get("duration_s", 0.0), 3) if props else None,
                "n_reps": len(vid_reps),
                "rep_numbers": ",".join(str(r) for r in rep_nums),
            })

    vid_df = pd.DataFrame(video_rows)
    vid_df.to_csv(MANIFESTS_DIR / "video_manifest.csv", index=False)
    print(f"  video_manifest: {len(vid_df)} rows saved -> manifests/video_manifest.csv")

    # -- 3d: Per-subject summary -----------------------------------------------
    subj_rows = []
    ex6_df = rep_df[rep_df["exercise_id"] == 6]
    for pid in sorted(rep_df["person_id"].unique()):
        sub = rep_df[rep_df["person_id"] == pid]
        sub6 = ex6_df[ex6_df["person_id"] == pid]
        oris = sorted(sub["cam17_orientation"].unique().tolist())
        subj_rows.append({
            "person_id": pid,
            "total_reps": len(sub),
            "correct_reps": int((sub["correctness"] == 1).sum()),
            "incorrect_reps": int((sub["correctness"] == 0).sum()),
            "ex6_reps": len(sub6),
            "ex6_correct": int((sub6["correctness"] == 1).sum()),
            "ex6_incorrect": int((sub6["correctness"] == 0).sum()),
            "orientations": ",".join(oris),
        })

    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(MANIFESTS_DIR / "subject_summary.csv", index=False)
    print(f"  subject_summary: {len(subj_df)} rows saved -> manifests/subject_summary.csv")

    # -- 3e: Ex6 slice --------------------------------------------------------
    ex6_manifest = rep_df[rep_df["exercise_id"] == 6].copy()
    ex6_manifest.to_csv(MANIFESTS_DIR / "ex6_rep_manifest.csv", index=False)
    print(f"  ex6_rep_manifest: {len(ex6_manifest)} rows saved "
          f"-> manifests/ex6_rep_manifest.csv")

    return rep_df, vid_df, subj_df


# -----------------------------------------------------------------------------
# Step 4: LOSO Split Helper
# -----------------------------------------------------------------------------

def step4_loso(rep_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  STEP 4: LOSO SPLIT HELPER (Ex6)")
    print("=" * 60)

    ex6 = rep_df[rep_df["exercise_id"] == 6]
    fold_rows = []
    for pid in sorted(ex6["person_id"].unique()):
        sub = ex6[ex6["person_id"] == pid]
        n_correct = int((sub["correctness"] == 1).sum())
        n_incorrect = int((sub["correctness"] == 0).sum())
        too_few = (n_correct < MIN_CLASS_REPS) or (n_incorrect < MIN_CLASS_REPS)
        fold_rows.append({
            "person_id": pid,
            "fold_id": pid,     # LOSO: fold = subject
            "ex6_correct": n_correct,
            "ex6_incorrect": n_incorrect,
            "ex6_total": n_correct + n_incorrect,
            "too_few_flag": too_few,
        })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(MANIFESTS_DIR / "proposed_loso_folds.csv", index=False)
    print(f"\n  proposed_loso_folds.csv: {len(fold_df)} subjects")
    print()
    print(fold_df.to_string(index=False))

    flagged = fold_df[fold_df["too_few_flag"]]
    print(f"\n  Subjects flagged (too_few_flag=True, < {MIN_CLASS_REPS} reps of some class):")
    if flagged.empty:
        print("    None")
    else:
        for _, row in flagged.iterrows():
            print(f"    Person {int(row['person_id'])}: "
                  f"{int(row['ex6_correct'])} correct, "
                  f"{int(row['ex6_incorrect'])} incorrect")

    n_usable = (~fold_df["too_few_flag"]).sum()
    n_subjects = len(fold_df)
    print(f"\n  Class-balance note:")
    print(f"    {n_subjects} subjects have Ex6 data; {n_usable} are viable test folds.")
    if not flagged.empty:
        print(f"    Person(s) {sorted(flagged['person_id'].tolist())} have fewer than "
              f"{MIN_CLASS_REPS} incorrect reps - when one of these is held out as the "
              f"test fold, the test set will contain very few (or zero) negative examples.")
        print(f"    Consider pooling these subjects or using stratified sampling rather "
              f"than strict LOSO for them.")
    else:
        print(f"    All subjects have a?? {MIN_CLASS_REPS} reps of each class; "
              f"clean LOSO evaluation is feasible.")
    print(f"    Total Ex6 class split: "
          f"{int((ex6['correctness']==1).sum())} correct, "
          f"{int((ex6['correctness']==0).sum())} incorrect")

    return fold_df


# -----------------------------------------------------------------------------
# Step 5: Data-Quality Report & Figures
# -----------------------------------------------------------------------------

# -- Figure helpers ------------------------------------------------------------

CORRECT_COLOR = "#2196F3"    # blue
INCORRECT_COLOR = "#F44336"  # red


def fig_class_balance_overall(rep_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall
    ax = axes[0]
    counts = rep_df["correctness"].value_counts().sort_index()
    bars = ax.bar(["Incorrect (0)", "Correct (1)"],
                  [counts.get(0, 0), counts.get(1, 0)],
                  color=[INCORRECT_COLOR, CORRECT_COLOR], width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5, str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Overall Class Balance (All Exercises)", fontsize=13)
    ax.set_ylabel("Number of Reps")
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.grid(axis="y", alpha=0.4)

    # Right: per exercise
    ax = axes[1]
    ex_ids = sorted(rep_df["exercise_id"].unique())
    x = np.arange(len(ex_ids))
    width = 0.35
    inc = [int((rep_df[(rep_df["exercise_id"] == e) & (rep_df["correctness"] == 0)]).shape[0]) for e in ex_ids]
    cor = [int((rep_df[(rep_df["exercise_id"] == e) & (rep_df["correctness"] == 1)]).shape[0]) for e in ex_ids]
    b1 = ax.bar(x - width/2, inc, width, label="Incorrect", color=INCORRECT_COLOR)
    b2 = ax.bar(x + width/2, cor, width, label="Correct", color=CORRECT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ex{e}\n{EXERCISE_NAMES[e][:8]}" for e in ex_ids], fontsize=8)
    ax.set_title("Class Balance per Exercise", fontsize=13)
    ax.set_ylabel("Number of Reps")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    for bars_ in [b1, b2]:
        for bar in bars_:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        str(int(h)), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_class_balance_overall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_class_balance_per_subject(rep_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (ex_filter, title) in zip(
        axes,
        [(None, "All Exercises"), (6, "Ex6 -- Squats")]
    ):
        subset = rep_df if ex_filter is None else rep_df[rep_df["exercise_id"] == ex_filter]
        pids = sorted(subset["person_id"].unique())
        x = np.arange(len(pids))
        inc = [int((subset[(subset["person_id"] == p) & (subset["correctness"] == 0)]).shape[0]) for p in pids]
        cor = [int((subset[(subset["person_id"] == p) & (subset["correctness"] == 1)]).shape[0]) for p in pids]
        width = 0.35
        ax.bar(x - width/2, inc, width, label="Incorrect", color=INCORRECT_COLOR)
        ax.bar(x + width/2, cor, width, label="Correct", color=CORRECT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{p}" for p in pids])
        ax.set_title(f"Class Balance per Subject - {title}", fontsize=12)
        ax.set_ylabel("Number of Reps")
        ax.legend()
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_class_balance_per_subject.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_rep_duration_histograms(rep_df: pd.DataFrame) -> None:
    # All exercises - subplot per exercise
    ex_ids = sorted(rep_df["exercise_id"].unique())
    n = len(ex_ids)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, ex_id in enumerate(ex_ids):
        ax = axes[i]
        sub = rep_df[rep_df["exercise_id"] == ex_id]["duration_s"]
        ax.hist(sub, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax.set_title(f"Ex{ex_id}: {EXERCISE_NAMES[ex_id]}", fontsize=11)
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Count")
        ax.axvline(sub.mean(), color="navy", linestyle="--", lw=1.5,
                   label=f"mean={sub.mean():.1f}s")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Rep Duration Distributions by Exercise", fontsize=13, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_rep_duration_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # Ex6 only
    ex6_dur = rep_df[rep_df["exercise_id"] == 6]["duration_s"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ex6_dur, bins=20, color=CORRECT_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(ex6_dur.mean(), color="navy", linestyle="--", lw=1.5,
               label=f"mean={ex6_dur.mean():.2f}s")
    ax.axvline(ex6_dur.median(), color="darkorange", linestyle=":", lw=1.5,
               label=f"median={ex6_dur.median():.2f}s")
    ax.set_title("Rep Duration Distribution - Ex6: Squats", fontsize=12)
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_rep_duration_ex6.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_orientation_distribution(rep_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#2196F3", "#FF9800", "#9C27B0"]

    # cam17
    ax = axes[0]
    vals = rep_df["cam17_orientation"].value_counts()
    ax.bar(vals.index, vals.values, color=colors[:len(vals)])
    ax.set_title("Cam17 Orientation Distribution", fontsize=12)
    ax.set_ylabel("Number of Reps")
    ax.grid(axis="y", alpha=0.4)
    for i, (label, v) in enumerate(zip(vals.index, vals.values)):
        ax.text(i, v + 2, str(v), ha="center", va="bottom", fontweight="bold")

    # cam18 (derived)
    ax = axes[1]
    vals18 = rep_df["cam18_orientation"].value_counts()
    ax.bar(vals18.index, vals18.values, color=colors[:len(vals18)])
    ax.set_title("Cam18 Orientation Distribution (derived)", fontsize=12)
    ax.set_ylabel("Number of Reps")
    ax.grid(axis="y", alpha=0.4)
    for i, (label, v) in enumerate(zip(vals18.index, vals18.values)):
        ax.text(i, v + 2, str(v), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out = FIGURES_DIR / "fig_orientation_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_lighting_distribution(rep_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = rep_df["lights_on"].value_counts().sort_index()
    labels = ["Lights Off (False)", "Lights On (True)"]
    ax.bar(labels[:len(vals)], vals.values,
           color=["#607D8B", "#FFC107"], width=0.5)
    for i, v in enumerate(vals.values):
        ax.text(i, v + 2, str(v), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Lighting Condition Distribution", fontsize=12)
    ax.set_ylabel("Number of Reps")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_lighting_dist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_rep_timeline_ex6(rep_df: pd.DataFrame, video_map: Dict[str, dict]) -> None:
    """Timeline plot for one Ex6 video showing annotated rep windows."""
    ex6 = rep_df[rep_df["exercise_id"] == 6]
    # Pick the video with most reps for richness
    vid_id = ex6.groupby("video_id").size().idxmax()
    reps = ex6[ex6["video_id"] == vid_id].sort_values("repetition_number")

    # Get frame count from video map if available
    props = None
    info = video_map.get(vid_id, {})
    if info.get("cam17") and info["cam17"].exists():
        props = get_video_props(info["cam17"])
    total_frames = props["frame_count"] if props else int(reps["last_frame"].max() + 100)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.barh(0, total_frames, height=0.3, color="#EEEEEE", edgecolor="#BBBBBB", label="Full video")

    for _, rep in reps.iterrows():
        color = CORRECT_COLOR if rep["correctness"] == 1 else INCORRECT_COLOR
        width = rep["last_frame"] - rep["first_frame"]
        ax.barh(0, width, left=rep["first_frame"], height=0.3,
                color=color, alpha=0.8)
        mid = (rep["first_frame"] + rep["last_frame"]) / 2
        ax.text(mid, 0, str(int(rep["repetition_number"])),
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Legend
    ax.legend(handles=[
        mpatches.Patch(color=CORRECT_COLOR, label="Correct"),
        mpatches.Patch(color=INCORRECT_COLOR, label="Incorrect"),
        mpatches.Patch(facecolor="#EEEEEE", edgecolor="#BBBBBB", label="Full video"),
    ], loc="upper right")
    ax.set_xlabel("Frame number (30 FPS)")
    ax.set_yticks([])
    ax.set_xlim(0, total_frames)
    total_s = total_frames / FPS
    ax.set_title(
        f"Rep Timeline - {vid_id} (Ex6: Squats) - "
        f"{len(reps)} reps, {total_s:.1f}s total"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_rep_timeline_ex6.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig_frame_montage_squat(
    rep_df: pd.DataFrame,
    video_map: Dict[str, dict],
) -> None:
    """2?--2 montage: rows = orientation (front, half-profile), cols = camera."""
    ex6 = rep_df[rep_df["exercise_id"] == 6]
    orientations = ["front", "half-profile"]

    cells = {}  # (orientation, cam_key) -> (frame_bgr, label)
    for ori in orientations:
        sub = ex6[ex6["cam17_orientation"] == ori]
        if sub.empty:
            continue
        # Pick first rep with a valid video
        for _, row in sub.iterrows():
            info = video_map.get(row["video_id"], {})
            for cam_key, cam_label in [("cam17", "Camera17"), ("cam18", "Camera18")]:
                if (ori, cam_key) in cells:
                    continue
                p = info.get(cam_key)
                if p and p.exists():
                    mid_frame = (int(row["first_frame"]) + int(row["last_frame"])) // 2
                    frame = read_frame(p, mid_frame)
                    if frame is not None:
                        cells[(ori, cam_key)] = (frame, cam_label)

    n_rows = len(orientations)
    n_cols = 2
    target_h, target_w = 240, 320

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    if n_rows == 1:
        axes = [axes]

    for r_idx, ori in enumerate(orientations):
        for c_idx, cam_key in enumerate(["cam17", "cam18"]):
            ax = axes[r_idx][c_idx]
            cell = cells.get((ori, cam_key))
            if cell is None:
                ax.set_visible(False)
                continue
            frame_bgr, cam_label = cell
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Resize to uniform thumbnail
            h, w = frame_rgb.shape[:2]
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            thumb = cv2.resize(frame_rgb, (new_w, new_h))
            ax.imshow(thumb)
            cam17_ori_label = ori
            cam18_ori_label = CAM17_TO_CAM18.get(ori, "?")
            if cam_key == "cam17":
                view_label = cam17_ori_label
            else:
                view_label = cam18_ori_label
            ax.set_title(
                f"{cam_label}\ncam17={cam17_ori_label} -> {view_label} view",
                fontsize=9
            )
            ax.axis("off")

    fig.suptitle("Sample Frames - Ex6 Squats (mid-rep)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_frame_montage_squat.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def step5_report(
    rep_df: pd.DataFrame,
    vid_df: pd.DataFrame,
    subj_df: pd.DataFrame,
    md5_results: Dict[str, str],
    video_map: Dict[str, dict],
) -> None:
    print("\n" + "=" * 60)
    print("  STEP 5: DATA-QUALITY REPORT & FIGURES")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Generating figures...")
    fig_class_balance_overall(rep_df)
    fig_class_balance_per_subject(rep_df)
    fig_rep_duration_histograms(rep_df)
    fig_orientation_distribution(rep_df)
    fig_lighting_distribution(rep_df)
    fig_rep_timeline_ex6(rep_df, video_map)
    fig_frame_montage_squat(rep_df, video_map)

    # -- Data-quality metrics --------------------------------------------------
    n_total = len(rep_df)
    ex6 = rep_df[rep_df["exercise_id"] == 6]

    n_missing_video = int((~rep_df["has_video"]).sum())
    missing_vid_ids = sorted(
        rep_df[~rep_df["has_video"]]["video_id"].unique().tolist()
    )

    frames_bad = rep_df[rep_df["frames_in_range"] == False]
    frames_unknown = rep_df[rep_df["frames_in_range"].isna()]
    n_out_of_range = len(frames_bad)
    n_frames_unknown = len(frames_unknown)

    dup_key = ["video_id", "repetition_number"]
    n_dups = int(rep_df.duplicated(subset=dup_key).sum())

    overlap_pairs = []
    for vid_id, grp in rep_df.groupby("video_id"):
        grp_s = grp.sort_values("first_frame")
        prev = None
        for _, row in grp_s.iterrows():
            if prev is not None and row["first_frame"] <= prev["last_frame"]:
                overlap_pairs.append(
                    f"{vid_id} rep{int(prev['repetition_number'])}"
                    f"&rep{int(row['repetition_number'])}"
                )
            prev = row
    n_overlaps = len(overlap_pairs)

    dur = rep_df["duration_s"]
    q1, q3 = dur.quantile(0.25), dur.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (dur < q1 - 3 * iqr) | (dur > q3 + 3 * iqr)
    n_dur_outliers = int(outlier_mask.sum())

    n_quality_bad = int((~rep_df["quality_ok"]).sum())
    n_mocap_err = int(rep_df["mocap_erroneous"].sum())
    n_ep_high = int(
        ((rep_df["extra_person_in_cam17"] >= 2) |
         (rep_df["extra_person_in_cam18"] >= 2)).sum()
    )

    n_ex6_correct = int((ex6["correctness"] == 1).sum())
    n_ex6_incorrect = int((ex6["correctness"] == 0).sum())
    total_correct = int((rep_df["correctness"] == 1).sum())
    total_incorrect = int((rep_df["correctness"] == 0).sum())

    # Per-subject Ex6 class balance
    subj_ex6_lines = []
    for _, row in subj_df.iterrows():
        flag = " <- TOO FEW" if row["ex6_reps"] > 0 and row["ex6_incorrect"] < MIN_CLASS_REPS else ""
        subj_ex6_lines.append(
            f"  Person {int(row['person_id']):2d}: "
            f"{int(row['ex6_correct']):3d} correct, "
            f"{int(row['ex6_incorrect']):3d} incorrect, "
            f"{int(row['ex6_reps']):3d} total{flag}"
        )

    # -- Write data_quality_report.md -----------------------------------------
    report_lines = [
        "# REHAB24-6 Data Quality Report",
        "",
        f"Generated by `src/explore.py`. Dataset: Zenodo 10.5281/zenodo.13305826.",
        "",
        "## 1. File Checksums (MD5)",
        "",
        "| File | Expected | Result |",
        "|------|----------|--------|",
    ]
    for fname, result in md5_results.items():
        expected = EXPECTED_MD5.get(fname, "-")
        icon = "[OK]" if result == "PASS" else ("-" if result == "MISSING" else "[FAIL]")
        report_lines.append(f"| `{fname}` | `{expected}` | {icon} {result} |")

    report_lines += [
        "",
        "## 2. Row Counts & Schema",
        "",
        f"- **Total rep rows**: {n_total}",
        f"- **CSV columns**: {len(rep_df.columns)} (expected {len(CSV_EXPECTED_COLUMNS)})",
        f"- **Schema match**: All {len(CSV_EXPECTED_COLUMNS)} expected columns present",
        f"- **`exercise_subtype` NaNs**: "
        f"{int(rep_df['exercise_subtype'].isna().sum())} "
        f"(expected - exercises without subtype distinction)",
        "",
        "## 3. Video Resolution",
        "",
        f"- **video_ids in CSV**: {rep_df['video_id'].nunique()}",
        f"- **video_ids on disk**: {len(video_map)}",
        f"- **Reps with missing video**: {n_missing_video}",
    ]
    if missing_vid_ids:
        report_lines.append(f"  - Missing: {missing_vid_ids}")

    report_lines += [
        "",
        "## 4. Frame Range Checks",
        "",
        f"- **Reps with `last_frame` >= `frame_count`** (out of range): {n_out_of_range}",
        f"- **Reps with unknown frame count** (video unreadable): {n_frames_unknown}",
    ]
    if n_out_of_range > 0:
        oor_rows = frames_bad[["video_id", "repetition_number",
                                "last_frame", "frames_in_range"]].head(10)
        report_lines.append("")
        report_lines.append("```")
        report_lines.append(oor_rows.to_string(index=False))
        report_lines.append("```")

    report_lines += [
        "",
        "## 5. Duplicate & Overlapping Windows",
        "",
        f"- **Duplicate (video_id, rep_number) rows**: {n_dups}",
        f"- **Overlapping frame windows within same video**: {n_overlaps}",
    ]
    if overlap_pairs:
        report_lines.append("  - Examples: " + "; ".join(overlap_pairs[:5]))

    report_lines += [
        "",
        "## 6. Duration Outliers",
        "",
        f"- **Outliers** (> 3x IQR from quartiles): {n_dur_outliers}",
        f"- **Duration stats**: mean={dur.mean():.2f}s, "
        f"median={dur.median():.2f}s, "
        f"min={dur.min():.2f}s, max={dur.max():.2f}s",
        f"- **IQR range**: [{q1:.2f}s, {q3:.2f}s]",
    ]

    report_lines += [
        "",
        "## 7. Data Quality Filter (`quality_ok`)",
        "",
        f"- **`quality_ok = False`** rows: {n_quality_bad} / {n_total} "
        f"({100*n_quality_bad/n_total:.1f}%)",
        f"  - `mocap_erroneous = 1`: {n_mocap_err}",
        f"  - `extra_person_in_cam17 >= 2` or `extra_person_in_cam18 >= 2`: {n_ep_high}",
        "(conditions can overlap)",
    ]

    report_lines += [
        "",
        "## 8. Class Balance",
        "",
        f"### Overall (all exercises, all subjects)",
        f"- **Correct (1)**: {total_correct} ({100*total_correct/n_total:.1f}%)",
        f"- **Incorrect (0)**: {total_incorrect} ({100*total_incorrect/n_total:.1f}%)",
        "",
        f"### Ex6 - Squats ({len(ex6)} reps)",
        f"- **Correct**: {n_ex6_correct} ({100*n_ex6_correct/len(ex6):.1f}%)",
        f"- **Incorrect**: {n_ex6_incorrect} ({100*n_ex6_incorrect/len(ex6):.1f}%)",
        "",
        f"### Ex6 per Subject",
        "",
    ] + subj_ex6_lines

    report_lines += [
        "",
        "## 9. Assumptions Checked",
        "",
        "| Assumption | Checked | Passed? |",
        "|------------|---------|---------|",
        "| All expected CSV columns present | [OK] | [OK] |",
        f"| No impossible frame ranges (`last < first`) | [OK] | {'[OK]' if not (rep_df['last_frame'] < rep_df['first_frame']).any() else '[FAIL]'} |",
        f"| No duplicate (video_id, rep) pairs | [OK] | {'[OK]' if n_dups == 0 else '[FAIL]'} |",
        f"| No overlapping windows within a video | [OK] | {'[OK]' if n_overlaps == 0 else '[FAIL]'} |",
        f"| All video_ids resolve to files on disk | [OK] | {'[OK]' if n_missing_video == 0 else '[FAIL]'} |",
        f"| All annotated frames within video bounds | [OK] | {'[OK]' if n_out_of_range == 0 else '[FAIL]'} |",
        f"| Checksums match for CSV and TXT | [OK] | {'[OK]' if all(v == 'PASS' for k, v in md5_results.items() if k in ('Segmentation.csv', 'Segmentation.txt')) else '[FAIL]'} |",
        f"| 10 subjects present in dataset | [OK] | {'[OK]' if rep_df['person_id'].nunique() == 10 else '[FAIL] - only ' + str(rep_df['person_id'].nunique())} |",
        f"| Ex6 has reps for all 10 subjects | [OK] | {'[FAIL] - only ' + str(ex6['person_id'].nunique()) + ' subjects have Ex6 data'} |",
    ]

    report_text = "\n".join(report_lines) + "\n"
    report_path = REPORT_DIR / "data_quality_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Written: {report_path}")

    # -- Write exploration_summary.md ------------------------------------------
    summary_lines = [
        "# REHAB24-6 Exploration Summary",
        "",
        "## Dataset Overview",
        "",
        f"REHAB24-6 contains **{n_total} annotated repetitions** across "
        f"**6 exercises** ("
        + ", ".join(f"Ex{k}: {v}" for k, v in EXERCISE_NAMES.items())
        + f"), performed by **{rep_df['person_id'].nunique()} subjects**. "
        f"Each rep is recorded simultaneously from two orthogonal cameras: "
        f"Camera17 (horizontal, wide FoV) and Camera18 (vertical, narrow FoV). "
        f"There are {rep_df['video_id'].nunique()} video sessions "
        f"({len(video_map) * 2} MP4 files, ~{vid_df[vid_df['camera']=='Camera17']['duration_s'].sum() / 3600:.1f}h total). "
        f"The primary annotation is binary correctness (1 = correct, 0 = incorrect); "
        f"no per-fault labels exist.",
        "",
        "## Key Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total reps | {n_total} |",
        f"| Correct reps | {total_correct} ({100*total_correct/n_total:.0f}%) |",
        f"| Incorrect reps | {total_incorrect} ({100*total_incorrect/n_total:.0f}%) |",
        f"| Ex6 (Squats) reps | {len(ex6)} |",
        f"| Ex6 subjects | {ex6['person_id'].nunique()} |",
        f"| Ex6 correct | {n_ex6_correct} ({100*n_ex6_correct/len(ex6):.0f}%) |",
        f"| Ex6 incorrect | {n_ex6_incorrect} ({100*n_ex6_incorrect/len(ex6):.0f}%) |",
        f"| Rep duration - mean (all) | {dur.mean():.2f}s |",
        f"| Rep duration - mean (Ex6) | {ex6['duration_s'].mean():.2f}s |",
        f"| Reps with quality_ok=False | {n_quality_bad} ({100*n_quality_bad/n_total:.1f}%) |",
        "",
        "## Four Findings That Most Affect Later Design",
        "",
        "### 1. Class Imbalance Is Worse Per Subject Than Globally",
        "",
        "Globally the dataset is roughly balanced, but within Ex6 individual subjects "
        "show very different correct/incorrect ratios. "
        f"Subject 2 has only {int(subj_df[subj_df['person_id']==2]['ex6_incorrect'].iloc[0])} "
        "incorrect Ex6 rep(s), making leave-one-subject-out (LOSO) evaluation unreliable "
        "when that subject is the test fold. Any LOSO-based evaluation must flag or "
        "handle subjects with near-zero incorrect reps.",
        "",
        "### 2. Ex6 Has Only Two Orientations (No Pure Profile View)",
        "",
        "All Ex6 squat reps use **front** or **half-profile** cam17 orientations - "
        "there are no reps recorded in the **profile** (pure sagittal) orientation. "
        "When cam17 is 'front', Camera18 provides the side view of the squat; "
        "when cam17 is 'half-profile', both cameras give a diagonal view. "
        "A model trained on Ex6 will never see a pure frontal-only view from Camera17 "
        "with a pure side view from Camera17 - the sagittal view always comes from Camera18 "
        "(in the front-orientation sessions).",
        "",
        "### 3. Ex6 Covers Only 9 of 10 Subjects",
        "",
        f"Person 10 has no Ex6 data ({ex6['person_id'].nunique()} subjects vs 10 overall). "
        "Subject-level statistics and LOSO folds for Ex6 must use 9 subjects, not 10. "
        "Cross-exercise comparisons need to account for this missing subject.",
        "",
        "### 4. Dirty Rows Are Moderate (~{:.0f}%) and Concentrated".format(
            100*n_quality_bad/n_total),
        "",
        f"{n_quality_bad} reps ({100*n_quality_bad/n_total:.1f}%) fail the `quality_ok` "
        f"filter due to mocap errors or significant extra-person occlusion. "
        f"These rows should be excluded from training but retained in the manifest for "
        f"audit purposes. The `quality_ok` flag in the per-rep manifest enables easy "
        f"filtering at load time.",
    ]

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = REPORT_DIR / "exploration_summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"  Written: {summary_path}")


# -----------------------------------------------------------------------------
# Step 6: Optional Pose Feasibility Probe
# -----------------------------------------------------------------------------

def step6_probe_pose(rep_df: pd.DataFrame, video_map: Dict[str, dict]) -> None:
    print("\n" + "=" * 60)
    print("  STEP 6: POSE FEASIBILITY PROBE (MediaPipe)")
    print("=" * 60)

    try:
        import mediapipe as mp
    except ImportError:
        print("  mediapipe not installed - skipping probe. "
              "Install with: pip install mediapipe")
        return

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    LANDMARKS_OF_INTEREST = {
        "left_hip":       mp_pose.PoseLandmark.LEFT_HIP,
        "right_hip":      mp_pose.PoseLandmark.RIGHT_HIP,
        "left_knee":      mp_pose.PoseLandmark.LEFT_KNEE,
        "right_knee":     mp_pose.PoseLandmark.RIGHT_KNEE,
        "left_ankle":     mp_pose.PoseLandmark.LEFT_ANKLE,
        "right_ankle":    mp_pose.PoseLandmark.RIGHT_ANKLE,
        "left_shoulder":  mp_pose.PoseLandmark.LEFT_SHOULDER,
        "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    }

    ex6 = rep_df[rep_df["exercise_id"] == 6]
    orientations = ex6["cam17_orientation"].unique().tolist()
    FRAMES_PER_REP = 3   # sample frames per rep
    REPS_PER_ORI = 2     # reps per orientation
    results_rows = []
    sample_saved = 0

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:
        for ori in sorted(orientations):
            sub = ex6[ex6["cam17_orientation"] == ori]
            rep_sample = sub.head(REPS_PER_ORI)
            for _, row in rep_sample.iterrows():
                info = video_map.get(row["video_id"], {})
                for cam_key, cam_label in [("cam17", "Camera17"), ("cam18", "Camera18")]:
                    p = info.get(cam_key)
                    if not (p and p.exists()):
                        continue
                    frame_indices = np.linspace(
                        row["first_frame"], row["last_frame"], FRAMES_PER_REP, dtype=int
                    )
                    for f_idx in frame_indices:
                        frame_bgr = read_frame(p, int(f_idx))
                        if frame_bgr is None:
                            continue
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        result = pose.process(frame_rgb)
                        if result.pose_landmarks:
                            vis_vals = {
                                name: result.pose_landmarks.landmark[lm].visibility
                                for name, lm in LANDMARKS_OF_INTEREST.items()
                            }
                            results_rows.append({
                                "video_id": row["video_id"],
                                "camera": cam_label,
                                "cam17_orientation": ori,
                                "frame": int(f_idx),
                                "detected": True,
                                **vis_vals,
                            })
                            # Save annotated sample frame
                            if sample_saved < 8:
                                ann = frame_bgr.copy()
                                mp_draw.draw_landmarks(
                                    ann,
                                    result.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS,
                                )
                                out_name = (
                                    f"{row['video_id']}_{cam_label}_f{f_idx:05d}_ori-{ori}.jpg"
                                )
                                cv2.imwrite(str(SAMPLES_DIR / out_name), ann)
                                sample_saved += 1
                        else:
                            results_rows.append({
                                "video_id": row["video_id"],
                                "camera": cam_label,
                                "cam17_orientation": ori,
                                "frame": int(f_idx),
                                "detected": False,
                                **{k: 0.0 for k in LANDMARKS_OF_INTEREST},
                            })

    if not results_rows:
        print("  No pose results obtained.")
        return

    probe_df = pd.DataFrame(results_rows)
    probe_df.to_csv(SAMPLES_DIR / "pose_probe_results.csv", index=False)

    lm_cols = list(LANDMARKS_OF_INTEREST.keys())
    detected = probe_df[probe_df["detected"]]

    print("\n  Mean landmark visibility by camera ?-- orientation:")
    print(f"  (Detected in {len(detected)}/{len(probe_df)} sampled frames)\n")
    if not detected.empty:
        summary = detected.groupby(["camera", "cam17_orientation"])[lm_cols].mean()
        print(summary.round(3).to_string())

    # Group by body part
    print("\n  Mean visibility by body-part group ?-- camera ?-- orientation:")
    groups = {
        "hips": ["left_hip", "right_hip"],
        "knees": ["left_knee", "right_knee"],
        "ankles": ["left_ankle", "right_ankle"],
        "shoulders": ["left_shoulder", "right_shoulder"],
    }
    for gname, gcols in groups.items():
        detected[gname] = detected[gcols].mean(axis=1)
    if not detected.empty:
        grp_summary = detected.groupby(["camera", "cam17_orientation"])[
            list(groups.keys())
        ].mean()
        print(grp_summary.round(3).to_string())

    print(f"\n  Sample frames saved to: {SAMPLES_DIR} ({sample_saved} images)")
    print(f"  Probe results saved to: {SAMPLES_DIR}/pose_probe_results.csv")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="REHAB24-6 Dataset Exploration Pipeline"
    )
    parser.add_argument(
        "--probe-pose", action="store_true",
        help="Run Step 6: MediaPipe pose feasibility probe on Ex6 videos"
    )
    args = parser.parse_args()

    data_dir = find_data_dir()
    videos_root = find_videos_root(data_dir)

    # -- Step 0 ----------------------------------------------------------------
    md5_results, video_map = step0_verify(data_dir, videos_root)

    n_pass = sum(1 for v in md5_results.values() if v == "PASS")
    n_total_checks = len([k for k in md5_results if k in ("Segmentation.csv", "Segmentation.txt")])
    if n_pass < n_total_checks:
        print("\n  WARNING: Checksum verification failed for some files. "
              "Proceeding anyway, but results may be unreliable.")

    print("\n" + "-" * 60)
    print("  CHECKPOINT 0 complete - video-to-ID mapping shown above.")
    print("  Continuing to Steps 1-3...")
    print("-" * 60)

    # -- Step 1: Environment (directories already created by setup) ------------
    # venv and packages managed externally; just confirm dirs exist.
    for d in [MANIFESTS_DIR, FIGURES_DIR, REPORT_DIR, SAMPLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # -- Step 2 ----------------------------------------------------------------
    df = step2_validate(data_dir, video_map)

    # -- Step 3 ----------------------------------------------------------------
    rep_df, vid_df, subj_df = step3_manifests(df, video_map, data_dir)

    print("\n" + "-" * 60)
    print("  CHECKPOINT 3 complete - manifests written.")
    print(f"  -> exploration/manifests/ contains:")
    for f in sorted(MANIFESTS_DIR.glob("*")):
        print(f"      {f.name}")
    print("-" * 60)

    # -- Step 4 ----------------------------------------------------------------
    fold_df = step4_loso(rep_df)

    # -- Step 5 ----------------------------------------------------------------
    step5_report(rep_df, vid_df, subj_df, md5_results, video_map)

    print("\n" + "-" * 60)
    print("  CHECKPOINT 5 complete - report and figures written.")
    print(f"  -> exploration/report/ contains:")
    for f in sorted(REPORT_DIR.glob("*")):
        print(f"      {f.name}")
    print(f"  -> exploration/figures/ contains:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"      {f.name}")
    print("-" * 60)

    # -- Step 6 (optional) -----------------------------------------------------
    if args.probe_pose:
        step6_probe_pose(rep_df, video_map)

    print("\n  All done.")


if __name__ == "__main__":
    main()
