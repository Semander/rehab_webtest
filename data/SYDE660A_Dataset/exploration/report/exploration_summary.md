# REHAB24-6 Exploration Summary

## Dataset Overview

REHAB24-6 contains **1072 annotated repetitions** across **6 exercises** (Ex1: Arm Abduction, Ex2: Arm VW, Ex3: Push-ups, Ex4: Leg Abduction, Ex5: Leg Lunge, Ex6: Squats), performed by **10 subjects**. Each rep is recorded simultaneously from two orthogonal cameras: Camera17 (horizontal, wide FoV) and Camera18 (vertical, narrow FoV). There are 65 video sessions (130 MP4 files, ~1.7h total). The primary annotation is binary correctness (1 = correct, 0 = incorrect); no per-fault labels exist.

## Key Statistics

| Metric | Value |
|--------|-------|
| Total reps | 1072 |
| Correct reps | 568 (53%) |
| Incorrect reps | 504 (47%) |
| Ex6 (Squats) reps | 195 |
| Ex6 subjects | 9 |
| Ex6 correct | 134 (69%) |
| Ex6 incorrect | 61 (31%) |
| Rep duration - mean (all) | 3.99s |
| Rep duration - mean (Ex6) | 3.31s |
| Reps with quality_ok=False | 283 (26.4%) |

## Four Findings That Most Affect Later Design

### 1. Class Imbalance Is Worse Per Subject Than Globally

Globally the dataset is roughly balanced, but within Ex6 individual subjects show very different correct/incorrect ratios. Subject 2 has only 1 incorrect Ex6 rep(s), making leave-one-subject-out (LOSO) evaluation unreliable when that subject is the test fold. Any LOSO-based evaluation must flag or handle subjects with near-zero incorrect reps.

### 2. Ex6 Has Only Two Orientations (No Pure Profile View)

All Ex6 squat reps use **front** or **half-profile** cam17 orientations - there are no reps recorded in the **profile** (pure sagittal) orientation. When cam17 is 'front', Camera18 provides the side view of the squat; when cam17 is 'half-profile', both cameras give a diagonal view. A model trained on Ex6 will never see a pure frontal-only view from Camera17 with a pure side view from Camera17 - the sagittal view always comes from Camera18 (in the front-orientation sessions).

### 3. Ex6 Covers Only 9 of 10 Subjects

Person 10 has no Ex6 data (9 subjects vs 10 overall). Subject-level statistics and LOSO folds for Ex6 must use 9 subjects, not 10. Cross-exercise comparisons need to account for this missing subject.

### 4. Dirty Rows Are Moderate (~26%) and Concentrated

283 reps (26.4%) fail the `quality_ok` filter due to mocap errors or significant extra-person occlusion. These rows should be excluded from training but retained in the manifest for audit purposes. The `quality_ok` flag in the per-rep manifest enables easy filtering at load time.
