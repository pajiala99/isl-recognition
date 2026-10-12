"""Central configuration: paths, feature layout and training hyper-parameters."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"          # INCLUDE zips / extracted videos (untrusted input)
DATA_VIDEOS = ROOT / "data" / "videos"    # extracted category folders
DATA_KP = ROOT / "data" / "keypoints"     # one .npz of landmarks per video
MODELS = ROOT / "models"                  # mediapipe .task bundles + trained checkpoints
ARTIFACTS = ROOT / "artifacts"            # metrics, plots, reports

HAND_TASK = MODELS / "hand_landmarker.task"
POSE_TASK = MODELS / "pose_landmarker_lite.task"

# Upper-body pose landmarks we keep (MediaPipe Pose indices).
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]   # nose, shoulders, elbows, wrists, hips
N_POSE = len(POSE_IDX)
N_HAND = 21

# Feature block sizes per frame (see features.py build_frame_features).
F_POSE = N_POSE * 3          # x, y, visibility  -> 27
F_HAND_GLOBAL = 2 * N_HAND * 2   # both hands, body-normalised x, y -> 84
F_HAND_LOCAL = 2 * N_HAND * 3    # both hands, wrist-local x, y, z  -> 126
F_FLAGS = 2                      # hand-present flags
F_BASE = F_POSE + F_HAND_GLOBAL + F_HAND_LOCAL + F_FLAGS   # 239
F_TOTAL = F_BASE * 2             # base + temporal delta -> 478

SEQ_LEN = 64                 # frames every clip is resampled to
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv", ".mpg", ".mts", ".m4v"}


@dataclass
class TrainConfig:
    seq_len: int = SEQ_LEN
    hidden: int = 192
    layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 32
    label_smoothing: float = 0.1
    patience: int = 15
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15
    augment: dict = field(default_factory=lambda: {
        "p_mirror": 0.3, "rot_deg": 10.0, "scale": 0.1,
        "shift": 0.05, "noise": 0.01, "p_frame_drop": 0.1, "speed": 0.2,
    })
