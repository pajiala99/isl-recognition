"""Signer-invariant feature engineering: raw landmarks -> fixed-length sequences."""
from __future__ import annotations

import numpy as np

from .config import POSE_IDX, N_HAND, F_BASE, SEQ_LEN

L_SHOULDER, R_SHOULDER = 11, 12


def _body_frame(pose: np.ndarray):
    """Shoulder-centred origin and shoulder-width scale, per frame.

    Makes features invariant to where the signer stands and how far they are
    from the camera - the single biggest nuisance factor in INCLUDE.
    """
    ls, rs = pose[:, L_SHOULDER, :2], pose[:, R_SHOULDER, :2]
    origin = (ls + rs) / 2.0
    scale = np.linalg.norm(ls - rs, axis=-1, keepdims=True)
    scale = np.where(scale < 1e-3, 1.0, scale)
    return origin, scale


def build_frame_features(pose, left, right, mask) -> np.ndarray:
    """(T, F_BASE) float32 feature matrix for one clip."""
    origin, scale = _body_frame(pose)                      # (T,2), (T,1)

    sel = pose[:, POSE_IDX, :]                             # (T, P, 4)
    pose_xy = (sel[..., :2] - origin[:, None, :]) / scale[:, None, :]
    pose_feat = np.concatenate([pose_xy, sel[..., 3:4]], -1).reshape(len(pose), -1)

    blocks = [pose_feat]
    local_blocks = []
    for hand, m in ((left, mask[:, 0]), (right, mask[:, 1])):
        # Global: where the hand is relative to the body (location of the sign).
        g = (hand[..., :2] - origin[:, None, :]) / scale[:, None, :]
        g *= m[:, None, None]
        blocks.append(g.reshape(len(hand), -1))

        # Local: hand shape only - wrist-centred, size-normalised (handshape).
        wrist = hand[:, 0:1, :]
        loc = hand - wrist
        hs = np.linalg.norm(loc[:, 9, :2], axis=-1, keepdims=True)   # wrist->middle MCP
        hs = np.where(hs < 1e-3, 1.0, hs)
        loc = loc / hs[:, None]
        loc *= m[:, None, None]
        local_blocks.append(loc.reshape(len(hand), -1))

    feats = np.concatenate(blocks + local_blocks + [mask], axis=-1).astype(np.float32)
    assert feats.shape[1] == F_BASE, (feats.shape, F_BASE)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def resample(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Linear-interpolate a variable-length clip onto a fixed time grid."""
    t = len(seq)
    if t == length:
        return seq.astype(np.float32)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, length)
    out = np.empty((length, seq.shape[1]), np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(dst, src, seq[:, c])
    return out


def with_deltas(seq: np.ndarray) -> np.ndarray:
    """Append frame-to-frame velocity: movement matters as much as posture in ISL."""
    d = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, d], axis=-1)


def pooled_stats(seq: np.ndarray) -> np.ndarray:
    """Fixed-size summary vector for the classical SVM / Random-Forest baselines."""
    return np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0)])
