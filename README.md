# rehab-AI

Offline squat assessment on the REHAB24-6 dataset: MediaPipe pose extraction,
joint-angle features, and interpretable correct/incorrect scoring with
plain-language feedback. SYDE 660A Group 5.

Key documents:

- [data/SYDE660A_Dataset/PLAN.md](data/SYDE660A_Dataset/PLAN.md) — overall pipeline plan (six modules, validation strategy)
- [data/SYDE660A_Dataset/REPORT.md](data/SYDE660A_Dataset/REPORT.md) — dataset exploration report and the four-stage next-steps plan
- [data/SYDE660A_Dataset/exploration/](data/SYDE660A_Dataset/exploration/) — figures, data-quality report, and the rep manifests that drive the pipeline

## Repo layout

```
src/rehab_ai/            main package: PoseEstimator, Pose, pose_extraction (Stage 1)
src/config/settings.py   paths to the pretrained MediaPipe models
models/pretrained/       MediaPipe pose_landmarker .task files (lite/full/heavy)
notebooks/               live-webcam pose demo notebooks
data/SYDE660A_Dataset/   REHAB24-6 dataset + exploration outputs (videos NOT in git)
data/poses/ex6/          cached per-rep pose landmarks (committed, ~30MB)
```

## Setup

Requires Python 3.8-3.11 (mediapipe 0.10.5 has no wheels for newer versions).
From the repo root:

```bash
py -3.8 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Getting the data

The small annotation files and all exploration outputs (manifests, figures,
reports) are committed, as are the extracted pose landmarks in
`data/poses/ex6/`. **If you only need to work on Stages 2-4 (angles, phase
segmentation, classification), you do not need the videos at all.**

The videos (~2.6GB) are only needed to re-run pose extraction. Download from
Zenodo (REHAB24-6, DOI 10.5281/zenodo.13305826, CC-BY-NC-4.0):

```bash
cd data/SYDE660A_Dataset
curl -L -o videos.zip "https://zenodo.org/records/13305826/files/videos.zip?download=1"
unzip videos.zip        # creates videos/Ex1 ... videos/Ex6
```

## Stage 1: pose extraction

Runs MediaPipe Pose (VIDEO mode) over every annotated Ex6 squat rep listed in
`exploration/manifests/ex6_rep_manifest.csv` and caches landmarks to disk.
Camera choice follows the exploration report: Camera18 for front-orientation
sessions (sagittal view); both cameras for half-profile sessions, with the
higher-visibility one marked primary.

```bash
# from the repo root (~10 min on CPU with the full model)
PYTHONPATH=src .venv/Scripts/python -m rehab_ai.pose_extraction --model full
```

Already-extracted reps are skipped, so an interrupted run resumes where it
left off. `--model lite|full|heavy` trades speed for accuracy.

Outputs:

- `data/poses/ex6/{video_id}_rep{NN}_{cam}.npz` — one file per rep per camera:
  - `image_landmarks` `(n_frames, 33, 4)`: normalized x, y, z, visibility
  - `world_landmarks` `(n_frames, 33, 4)`: metric x, y, z, visibility
  - `frame_indices` plus video `width`/`height`/`fps` metadata
  - frames where detection failed are NaN
- `exploration/manifests/ex6_rep_manifest_pose.csv` — the rep manifest plus
  `pose_camera`, `pose_path`, per-knee visibility, and a `pose_ok` flag
  (false only when *both* knees are poorly visible; in a sagittal view the
  far knee is expected to be occluded).

Load a rep in two lines:

```python
import numpy as np
rep = np.load("data/poses/ex6/PM_008_rep01_cam18.npz")
```

## Live webcam demo

`notebooks/pose_estimator_test.ipynb` runs the `PoseEstimator` in LIVE_STREAM
mode against your webcam using the same pretrained models.

## Next steps (see REPORT.md for detail)

1. ~~Pose extraction~~ (done — this repo)
2. Joint angle computation (knee, hip, trunk) from cached landmarks
3. Phase segmentation within each rep (descent / bottom / ascent)
4. Interpretable classifier + feedback generation

## Citation / license

REHAB24-6: Cernek, Sedmidubsky, Budikova — SISAP 2024.
Dataset licensed CC-BY-NC-4.0 (academic, non-commercial).
