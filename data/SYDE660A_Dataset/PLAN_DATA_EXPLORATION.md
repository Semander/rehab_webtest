# REHAB24-6 Dataset Exploration and Processing Plan

SYDE 660 Group 5. Owner: Michael (dataset and clinical pathway).
Scope: acquire, verify, understand, and pre-process REHAB24-6 into analysis-ready artifacts. This stops before angles, segmentation logic, scoring, and feedback, which are the separate full-pipeline plan.

## 1. Goal

Produce a trustworthy picture of what is in the dataset and a clean manifest that the modeling pipeline can load without re-parsing raw files. Surface anything that would affect later design choices: class balance per subject, usable camera views, frame-alignment assumptions, and dirty rows.

## 2. Data to acquire

REHAB24-6, Zenodo DOI 10.5281/zenodo.13305826, CC-BY-NC-4.0 (academic noncommercial). For exploration the essential files are `videos.zip` and `Segmentation.csv` plus `Segmentation.txt`. The joint and marker zips are only needed for the later pose-accuracy study.

Published md5 checksums to verify downloads:
- videos.zip: ed183a245a1b171638e422f7a288e5a8
- Segmentation.csv: 90b8fbd7445dd050bf27b17126c78fbe
- Segmentation.txt: 5f2a5b886c6f794f03e1a8f642738c86

Segmentation.csv columns:
`video_id, repetition_number, exercise_id, person_id, first_frame, last_frame, cam17_orientation, mocap_erroneous, exercise_subtype, lights_on, extra_person_in_cam17, extra_person_in_cam18, correctness`

Facts to rely on:
- 30 FPS, frame indices on that timeline.
- 6 exercises, squats are Ex6. 10 subjects.
- correctness is binary, 1 correct and 0 incorrect, no per-fault labels.
- Two cameras, Camera17 horizontal and Camera18 vertical, orthogonal to each other.

## 3. What to produce

Three kinds of output: a clean manifest, a data-quality report, and a set of figures.

### 3.1 Manifests (the main processing deliverable)

Per-rep manifest, one row per repetition, joining the annotations with derived and resolved fields:
- identifiers: video_id, repetition_number, exercise_id, exercise_name, person_id
- timing: first_frame, last_frame, n_frames, duration_s (n_frames divided by FPS)
- view: cam17_orientation, derived cam18_orientation, and which camera gives the side view for sagittal angles
- conditions: exercise_subtype, lights_on
- quality flags: mocap_erroneous, extra_person_in_cam17, extra_person_in_cam18, and a single quality_ok boolean
- label: correctness
- files: resolved video paths per camera, has_video boolean, frames_in_range boolean (last_frame within the actual video frame count)

Also produce a per-video manifest (resolution, fps, frame count, duration, exercises present) and a per-subject summary (rep counts, correct vs incorrect counts overall and for Ex6, orientations present). Save an Ex6-only slice of the rep manifest as its own file.

### 3.2 Split helper

Because there are only 10 subjects and later validation must be subject-wise, produce a per-subject table of correct vs incorrect counts for Ex6, flag any subject too imbalanced or too small to sit in a test fold alone, and write a proposed leave-one-subject-out fold list. This is planning input, not a final decision.

### 3.3 Data-quality report

A short markdown report covering: row counts, schema match against Segmentation.txt, missing or unresolved videos, reps whose frame range exceeds the video length, duplicate or overlapping rep windows within a video, duration outliers (very short or very long reps), NaNs or unexpected values, counts dropped by the quality flags, and class balance overall and per subject. State assumptions checked and any that failed.

### 3.4 Figures

Class balance overall and per subject, rep-duration histograms (all and Ex6), orientation distribution, lighting distribution, a rep-timeline plot for one example video showing the annotated rep windows, and a small montage of sample frames from each camera and orientation for the squat.

## 4. Optional feasibility probe

A small, clearly-bounded pose check, not the pipeline. Run MediaPipe on a handful of sampled frames from a few Ex6 videos across orientations and report mean visibility of hips, knees, ankles, and shoulders by view, with a few annotated sample frames saved. This answers whether MediaPipe can see the relevant joints in this data and which view is most usable, before any pipeline work starts. Keep it to a sample, do not build a cached pose store here.

## 5. Handoff

The per-rep manifest and the Ex6 slice are the artifacts the full pipeline consumes. The pipeline then adds pose extraction, angles, segmentation, scoring, and feedback on top, reading rep boundaries and labels from the manifest rather than re-parsing the raw CSV.

## 6. Out of scope here

Joint-angle computation, repetition segmentation algorithms, the scorer, and the feedback layer. Those are the separate full-pipeline plan and prompt.
