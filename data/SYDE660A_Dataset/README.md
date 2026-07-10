# REHAB24-6 Dataset Exploration

Exploration and pre-processing pipeline for the REHAB24-6 rehabilitation exercise dataset
(Zenodo [10.5281/zenodo.13305826](https://doi.org/10.5281/zenodo.13305826), CC-BY-NC-4.0).

**Scope:** data acquisition, verification, validation, manifests, and quality reporting only.
No joint-angle computation, segmentation, scoring, or feedback.

## Dataset Quick Facts

- 6 exercises: Ex1 Arm Abduction, Ex2 Arm VW, Ex3 Push-ups, Ex4 Leg Abduction, Ex5 Leg Lunge, Ex6 Squats
- 10 subjects, 30 FPS video, ~1 072 annotated repetitions
- Two orthogonal cameras: Camera17 (horizontal) and Camera18 (vertical)
- Binary correctness labels (1 = correct, 0 = incorrect); no per-fault labels

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

## Data Layout

Place raw data under `data/` (or keep at project root — the script auto-detects):

```
data/
  Segmentation.csv
  Segmentation.txt
  videos.zip           # original archive (optional, for checksum)
  videos/
    Ex1/  Ex2/  Ex3/  Ex4/  Ex5/  Ex6/
```

## Usage

```bash
# Full pipeline (Steps 0–5)
python -m src.explore

# Include Step 6: MediaPipe pose feasibility probe
python -m src.explore --probe-pose
```

## Outputs

```
exploration/
  manifests/
    rep_manifest.parquet       # primary handoff artifact — one row per rep
    rep_manifest.csv
    video_manifest.csv         # one row per MP4 file with OpenCV properties
    subject_summary.csv        # per-subject rep counts and Ex6 breakdown
    ex6_rep_manifest.csv       # rep_manifest filtered to squats
    proposed_loso_folds.csv    # LOSO fold assignments for Ex6
  figures/
    fig_class_balance_overall.png
    fig_class_balance_per_subject.png
    fig_rep_duration_all.png
    fig_rep_duration_ex6.png
    fig_orientation_dist.png
    fig_lighting_dist.png
    fig_rep_timeline_ex6.png
    fig_frame_montage_squat.png
  report/
    data_quality_report.md
    exploration_summary.md
  samples/                     # annotated frames from --probe-pose (optional)
```

## Steps

| Step | What it does |
|------|-------------|
| 0 | Verify MD5 checksums; discover video→ID mapping at runtime |
| 1 | Environment setup (venv + packages) |
| 2 | Load CSV, validate schema, flag NaNs / bad frames / overlaps |
| 3 | Build per-rep, per-video, per-subject manifests + Ex6 slice |
| 4 | Proposed LOSO folds for Ex6 with class-balance flags |
| 5 | Data-quality report (Markdown) + 7 figures |
| 6 | Optional MediaPipe pose visibility probe (`--probe-pose`) |
