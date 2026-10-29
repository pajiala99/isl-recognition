"""MediaPipe Tasks wrapper: video frames -> raw pose/hand landmark arrays."""
from __future__ import annotations

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

from .config import HAND_TASK, POSE_TASK, N_HAND


class quiet_stderr:
    """Silence native (C++) stderr for the duration of a block.

    glog and TFLite write straight to file descriptor 2, so Python-level logging
    settings cannot reach them. We only wrap model construction, where the noise is -
    never the processing loop, so real Python errors still surface normally.
    """

    def __enter__(self):
        import os
        self._saved = os.dup(2)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *exc):
        import os
        os.dup2(self._saved, 2)
        os.close(self._null)
        os.close(self._saved)
        return False


class LandmarkExtractor:
    """Runs hand + upper-body pose landmarkers over a clip.

    Returns, per clip:
        pose  (T, 33, 4)  x, y, z, visibility  (normalised image coords)
        left  (T, 21, 3)  x, y, z
        right (T, 21, 3)
        mask  (T, 2)      1.0 where that hand was detected
    """

    def __init__(self, min_conf: float = 0.4, running_mode: str = "video"):
        mode = RunningMode.VIDEO if running_mode == "video" else RunningMode.IMAGE
        with quiet_stderr():
            self._build(mode, min_conf)
        self.mode = mode
        self._last_ts = -1          # MediaPipe VIDEO mode demands strictly increasing stamps
        self._warm = False          # first inference also logs; silence that one call

    def _build(self, mode, min_conf):
        self.hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_TASK)),
            running_mode=mode, num_hands=2,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        ))
        self.pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_TASK)),
            running_mode=mode, num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        ))

    def close(self):
        self.hands.close()
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _detect(self, image, ts_ms: int):
        if self.mode == RunningMode.VIDEO:
            return (self.hands.detect_for_video(image, ts_ms),
                    self.pose.detect_for_video(image, ts_ms))
        return self.hands.detect(image), self.pose.detect(image)

    def _frame(self, rgb: np.ndarray, ts_ms: int):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == RunningMode.VIDEO:
            # one extractor may be reused across clips, whose stamps restart at 0
            ts_ms = max(int(ts_ms), self._last_ts + 1)
            self._last_ts = ts_ms
        if not self._warm:
            with quiet_stderr():          # graph warm-up logs on the first inference
                hres, pres = self._detect(image, ts_ms)
            self._warm = True
        else:
            hres, pres = self._detect(image, ts_ms)

        pose = np.zeros((33, 4), np.float32)
        if pres.pose_landmarks:
            for i, lm in enumerate(pres.pose_landmarks[0]):
                pose[i] = (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0) or 0.0)

        hands = np.zeros((2, N_HAND, 3), np.float32)
        mask = np.zeros(2, np.float32)
        for lms, hd in zip(hres.hand_landmarks, hres.handedness):
            # MediaPipe labels handedness from the camera's view; the mirror-image
            # label is the signer's actual hand, so "Left" -> signer's right hand.
            slot = 1 if hd[0].category_name == "Left" else 0
            if mask[slot]:                       # keep the more confident duplicate
                continue
            hands[slot] = [(lm.x, lm.y, lm.z) for lm in lms]
            mask[slot] = 1.0
        return pose, hands[0], hands[1], mask

    def run_video(self, path, stride: int = 1, max_frames: int | None = None):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        poses, lefts, rights, masks = [], [], [], []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                p, l, r, m = self._frame(rgb, int(idx / fps * 1000))
                poses.append(p); lefts.append(l); rights.append(r); masks.append(m)
                if max_frames and len(poses) >= max_frames:
                    break
            idx += 1
        cap.release()
        if not poses:
            raise ValueError(f"no frames decoded from {path}")
        return (np.stack(poses), np.stack(lefts), np.stack(rights),
                np.stack(masks), float(fps))
