from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class ModelTypePath():
    LITE = str(ROOT / "models" / "pretrained" / "pose_landmarker_lite.task")
    FULL = str(ROOT / "models" / "pretrained" / "pose_landmarker_full.task")
    HEAVY = str(ROOT / "models" / "pretrained" / "pose_landmarker_heavy.task")