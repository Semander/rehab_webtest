# Squat Segmentation and Rule-Based Scoring Plan (REHAB24-6, Ex6)

SYDE 660 Group 5. Owner: Michael (dataset and clinical pathway, validation support).
Target: offline squat assessment on held-out subjects that returns interpretable, user-facing feedback, validated against REHAB24-6 binary correctness labels.

## 1. Dataset

REHAB24-6, Zenodo DOI 10.5281/zenodo.13305826, license CC-BY-NC-4.0 (academic noncommercial). Cite Cernek, Sedmidubsky, Budikova, SISAP 2024.

Download (squat task needs only the first two; grab the joint zips later for R3/R4):

```bash
curl -L -o videos.zip        "https://zenodo.org/records/13305826/files/videos.zip?download=1"
curl -L -o Segmentation.csv  "https://zenodo.org/records/13305826/files/Segmentation.csv?download=1"
curl -L -o Segmentation.txt  "https://zenodo.org/records/13305826/files/Segmentation.txt?download=1"
# later, for pose-accuracy validation against ground truth:
curl -L -o 2d_joints.zip     "https://zenodo.org/records/13305826/files/2d_joints.zip?download=1"
curl -L -o 3d_joints.zip     "https://zenodo.org/records/13305826/files/3d_joints.zip?download=1"
curl -L -o joints_names.txt  "https://zenodo.org/records/13305826/files/joints_names.txt?download=1"
```

Contents that matter:
- 65 recordings, 30 FPS, two cameras (Camera17 horizontal wide FoV, Camera18 vertical narrow FoV).
- 6 exercises. Squats are exercise_id Ex6. (Ex1 Arm abduction, Ex2 Arm VW, Ex3 Push-ups, Ex4 Leg abduction, Ex5 Leg lunge.)
- 10 subjects, each doing reps both correctly and incorrectly, with subject-specific mistakes.
- Segmentation.csv: one row per repetition, with start/end frame and a binary correctness label.

Segmentation.csv columns:
`video_id, repetition_number, exercise_id, person_id, first_frame, last_frame, cam17_orientation, mocap_erroneous, exercise_subtype, lights_on, extra_person_in_cam17, extra_person_in_cam18, correctness`

Key points:
- Frame indices are on the 30 FPS timeline, so MediaPipe run per frame on the same video lines up directly with first_frame and last_frame.
- correctness is binary: 1 correct, 0 incorrect. There are no per-fault labels.
- cam17_orientation tells you the view (front, half-profile, profile). Camera18 is orthogonal, so front on cam17 maps to side on cam18 and vice versa.
- mocap_erroneous and extra_person flags let you drop dirty rows.

## 2. Scope for this task

- Exercise: Ex6 squats only.
- Camera and view: pick one camera and prefer the sagittal (side/profile) view for knee and hip flexion, since depth and trunk lean read cleanly from the side. Use cam17_orientation to select rows that give a side view. Note the choice and keep it consistent. Valgus and asymmetry read better from the front, so record which view each feedback item depends on.
- Pose source: re-extract with MediaPipe Pose from RGB. Do not use the OptiTrack skeletons in the scoring pipeline. They are ground truth for the separate R3/R4 accuracy study only.
- Label used for scoring: the binary correctness column.
- Output: a binary correct/incorrect decision plus interpretable, ranked, plain-language feedback per rep, plus a per-session summary.

## 3. Pipeline

Six modules, built and tested independently, then chained. This isolates failures and matches the report's component-then-integration testing plan.

1. Pose extraction. Run MediaPipe Pose on each Ex6 video, cache per-frame 33-landmark coordinates plus visibility to disk so you never re-run it. Save both image-space (x, y in pixels) and world landmarks.
2. Angle computation. From cached landmarks compute per-frame knee angle (left, right, mean), hip angle, and trunk inclination from vertical.
3. Segmentation. Detect rep boundaries from the mean knee-angle signal (smooth, then find squat-bottom minima with scipy find_peaks). Validate detected boundaries against Segmentation.csv first_frame and last_frame using temporal IoU and matched-boundary frame error. Then segment each rep into intra-rep phases (standing, descent, bottom, ascent) so feedback can be localized to where a fault happens.
4. Feature extraction. For each rep (use GT boundaries for the scoring evaluation so segmentation error does not contaminate it), compute global features (min knee angle, knee ROM, peak trunk lean, asymmetry, duration, descent/ascent balance) and keyframe-localized features measured at the bottom of the squat and per phase.
5. Rule-based scorer. Per rep, evaluate interpretable threshold checks. Emit a structured result: the binary label, and for each rule the pass/fail, the phase it applies to, the measured value, the target, the deviation magnitude, and the direction. Tune thresholds on the development split only.
6. Feedback generation. Turn the scorer's structured deviations into ranked, plain-language coaching tied to the phase where the fault occurred and the size of the deviation. Produce per-rep messages and a per-session summary.

## 4. User-facing interpretable feedback

This is the layer that replaces a bare correct/incorrect with usable guidance. It reuses the scorer's per-rule output rather than adding a separate model.

Intra-rep phase segmentation. Within a rep the mean knee angle goes high (standing) to low (bottom of squat) to high (standing). Define: descent from rep start to the knee-angle minimum, bottom as a short window around that minimum, ascent from the minimum to rep end. These phases are the landmark points where checks are applied.

Keyframe checks. Each fault is checked at the phase where it shows up:
- Insufficient depth: minimum knee angle at the bottom.
- Excessive forward lean: trunk inclination at the bottom.
- Knee valgus (front view only): inter-knee over inter-ankle distance at the bottom.
- Left-right asymmetry: max knee-angle difference across the rep, flagged at the phase where it peaks.
- Tempo or control: rep duration and descent/ascent balance across descent and ascent.
- Incomplete lockout (optional): knee angle at standing-end well below the standing-start maximum.

Fault to message mapping. Each fault maps to a template that names what happened, where in the movement, the measured value, and what to aim for. For example, an insufficient-depth fault becomes a message reporting the knee angle reached at the bottom and the target to beat. Messages are coaching language, not diagnosis.

Ranking and restraint. Each fault carries a deviation magnitude (how far past threshold). Rank worst first and show only the top one or two faults per rep so the user is not buried. Faults within a small margin of threshold are suppressed.

Session summary. Aggregate across the reps in a video: the most frequent fault, whether reps are improving or worsening across the set, and one short coaching paragraph.

Clinical-safety framing. Feedback is decision-support, not autonomous clinical guidance, matching the ethical position in the report's impact assessment. Templates avoid diagnostic claims, are stored in config so a clinician can review and edit them, and the output carries a short disclaimer.

## 5. Validation

- Split by subject (person_id), never by repetition. Use leave-subjects-out: tune thresholds on a development set of subjects, evaluate on held-out subjects. Leave-one-subject-out cross-validation across the 10 subjects is the cleanest version to report.
- Primary validated metric: classification accuracy on held-out subjects against the binary correctness label. Report precision, recall, F1, and a confusion matrix alongside it.
- Report segmentation boundary error separately from scoring accuracy.
- Feedback quality (optional but strengthens the report): the per-fault messages are interpretation and are not validated by the binary label alone. To estimate their quality, manually annotate the dominant fault on a small held-out sample of incorrect reps and report how often the system's top-ranked fault matches the human annotation. Treat this as a secondary, clearly-labelled metric.
- Drop reps flagged mocap_erroneous or with high extra_person values before evaluation, and state how many you dropped.

## 6. Honest caveats to carry into the report

- Binary labels only. The validated claim is correct vs incorrect. The feedback layer is interpretation built from interpretable features, not a validated fault classifier, unless you add the manual fault annotation above.
- Incorrect reps mix several mistake types across subjects, so a fixed rule set has a ceiling. Report which fault features actually separate the classes.
- Single view limits what you can measure. Sagittal angles need a side view; knee valgus and asymmetry read best from the front. Be explicit about which view each feedback item came from.
- The pipeline runs offline on pre-recorded video. Real-time smartphone inference stays a stretch goal, matching the committed deliverable in the report.

## 7. Mapping to report requirements

- R1 RGB only: satisfied, MediaPipe on RGB video, no depth or markers in the pipeline.
- R2 1080p at 24 FPS: measure MediaPipe throughput during extraction and report it. Offline scoring is the committed fallback if real-time is not reached.
- R3 pose accuracy: separate study, compare MediaPipe knee and hip angles against angles computed from the OptiTrack joints (map LeftUpLeg/LeftLeg/LeftFoot and right side to MediaPipe hip/knee/ankle), report mean joint-angle error.
- R4 pose precision: standard deviation of joint-angle error across each rep.
- F3 feedback function: the feedback layer satisfies the report's requirement to provide interpretable, text-based feedback rather than an opaque score.
