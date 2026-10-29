"""Watch the preprocessing happen on a real clip - the demo to show instead of slides.

Four panels, in sync, frame by frame:
  1. the raw video with the landmarks MediaPipe found drawn on top
  2. the same skeleton after body-frame normalisation, on its own axes
  3. the 239-number feature vector for the current frame, as a colour strip
  4. live numbers: shoulder width in pixels, scale divisor, hands detected

    python -m isl.visualize --word Tea                 # a held-out clip
    python -m isl.visualize --video path/to.mp4
    python -m isl.visualize --word Tea --save out.mp4  # record it as a fallback
    python -m isl.visualize --invariance --word Tea    # the invariance proof
    python -m isl.visualize --word Tea --figure stages.png

In --invariance mode the clip is shown twice, once zoomed in and shifted, and the two
normalised skeletons come out on top of each other - that is the whole argument for
shoulder-frame normalisation, visible in one window.

Keys: space = pause/resume, q = quit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import POSE_IDX, ROOT
from .features import build_frame_features

VIDEOS = ROOT / "data" / "videos" / "Video_Dataset" / "Video_Dataset"
TEAL, AMBER, GREY, INK = (120, 160, 15), (36, 158, 245), (150, 150, 150), (40, 30, 25)
PANEL_BG = (248, 246, 243)

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
# indices into POSE_IDX = [nose, L_sh, R_sh, L_el, R_el, L_wr, R_wr, L_hip, R_hip]
POSE_EDGES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8)]
L_SH, R_SH = 11, 12


def pick_clip(word: str | None, video: Path | None) -> Path:
    if video:
        return video
    folders = sorted(p for p in VIDEOS.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit(f"no videos under {VIDEOS}")
    if word:
        match = [f for f in folders if f.name.lower() == word.strip().lower()]
        if not match:
            raise SystemExit(f"'{word}' not found. Available: {', '.join(f.name for f in folders[:8])} ...")
        folder = match[0]
    else:
        folder = folders[0]
    clips = sorted(p for p in folder.glob("*.mp4") if "tilt" not in p.name)
    if not clips:
        raise SystemExit(f"no clips in {folder}")
    return clips[0]


def warp(frame, zoom: float, dx: int, dy: int):
    """Fake a different camera: zoom in and shift the signer off-centre."""
    h, w = frame.shape[:2]
    M = np.float32([[zoom, 0, dx + (1 - zoom) * w / 2], [0, zoom, dy + (1 - zoom) * h / 2]])
    return cv2.warpAffine(frame, M, (w, h), borderValue=(255, 255, 255))


def caption(panel, text):
    """Dark bar so the label stays readable over any frame."""
    w = panel.shape[1]
    cv2.rectangle(panel, (0, 0), (w, 38), (30, 26, 22), -1)
    cv2.putText(panel, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1,
                cv2.LINE_AA)
    return panel


def draw_landmarks(canvas, pose, left, right, mask):
    """Panel 1: what MediaPipe found, drawn on the real frame."""
    h, w = canvas.shape[:2]
    px = lambda p: (int(p[0] * w), int(p[1] * h))
    sel = pose[POSE_IDX]
    for a, b in POSE_EDGES:
        if sel[a, 3] > 0.3 and sel[b, 3] > 0.3:
            cv2.line(canvas, px(sel[a]), px(sel[b]), TEAL, 2)
    for i, p in enumerate(sel):
        if p[3] > 0.3:
            cv2.circle(canvas, px(p), 4, TEAL, -1)
    for hand, m in ((left, mask[0]), (right, mask[1])):
        if m < 0.5:
            continue
        for a, b in HAND_EDGES:
            cv2.line(canvas, px(hand[a]), px(hand[b]), AMBER, 2)
        for p in hand:
            cv2.circle(canvas, px(p), 3, (60, 60, 230), -1)
    if pose[L_SH, 3] > 0.3 and pose[R_SH, 3] > 0.3:
        a, b = px(pose[L_SH]), px(pose[R_SH])
        cv2.line(canvas, a, b, (230, 90, 40), 3)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.circle(canvas, mid, 6, (230, 90, 40), -1)
    return canvas


def draw_normalised(size, pose, left, right, mask, title="normalised (body frame)"):
    """Panel 2: the same skeleton after (x - shoulder_mid) / shoulder_width."""
    h, w = size
    canvas = np.full((h, w, 3), PANEL_BG, np.uint8)
    cx, cy, unit = w // 2, int(h * 0.42), int(min(w, h) * 0.30)

    for g in range(-3, 4):                                  # reference grid, 1 unit = shoulder width
        cv2.line(canvas, (cx + g * unit, 0), (cx + g * unit, h), (228, 224, 220), 1)
        cv2.line(canvas, (0, cy + g * unit), (w, cy + g * unit), (228, 224, 220), 1)
    cv2.line(canvas, (0, cy), (w, cy), (205, 200, 195), 1)
    cv2.line(canvas, (cx, 0), (cx, h), (205, 200, 195), 1)

    ls, rs = pose[L_SH, :2], pose[R_SH, :2]
    origin = (ls + rs) / 2.0
    scale = float(np.linalg.norm(ls - rs)) or 1.0
    if scale < 1e-3:
        scale = 1.0
    to_px = lambda p: (int(cx + (p[0] - origin[0]) / scale * unit),
                       int(cy + (p[1] - origin[1]) / scale * unit))

    sel = pose[POSE_IDX]
    for a, b in POSE_EDGES:
        if sel[a, 3] > 0.3 and sel[b, 3] > 0.3:
            cv2.line(canvas, to_px(sel[a]), to_px(sel[b]), TEAL, 2)
    for p in sel:
        if p[3] > 0.3:
            cv2.circle(canvas, to_px(p), 4, TEAL, -1)
    for hand, m in ((left, mask[0]), (right, mask[1])):
        if m < 0.5:
            continue
        for a, b in HAND_EDGES:
            cv2.line(canvas, to_px(hand[a]), to_px(hand[b]), AMBER, 2)
        for p in hand:
            cv2.circle(canvas, to_px(p), 3, (60, 60, 230), -1)

    cv2.line(canvas, (cx - unit // 2, cy), (cx + unit // 2, cy), (230, 90, 40), 3)
    cv2.putText(canvas, "1.0 = shoulder width", (cx - unit // 2, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 90, 40), 1)
    cv2.putText(canvas, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, INK, 1)
    return canvas


def draw_features(size, history, feats):
    """Panel 3: the feature matrix building up, one column per frame."""
    h, w = size
    canvas = np.full((h, w, 3), PANEL_BG, np.uint8)
    cv2.putText(canvas, "feature matrix: 239 numbers per frame", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, INK, 1)
    top, bottom = 38, h - 28
    if history:
        mat = np.stack(history, axis=1)                      # (239, T)
        norm = np.clip((mat + 2.0) / 4.0, 0, 1)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        img = cv2.resize(img, (w - 28, bottom - top), interpolation=cv2.INTER_NEAREST)
        canvas[top:bottom, 14:w - 14] = img
    cv2.putText(canvas, "time ->", (14, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)
    cv2.putText(canvas, f"{len(history)} frames kept", (w - 150, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)
    return canvas


def draw_stats(size, lines):
    h, w = size
    canvas = np.full((h, w, 3), PANEL_BG, np.uint8)
    cv2.putText(canvas, "what the numbers are doing", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, INK, 1)
    y = 52
    for label, value, colour in lines:
        cv2.putText(canvas, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, GREY, 1)
        cv2.putText(canvas, value, (w // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        y += 26
    return canvas


def tile(panels, cell):
    """2x2 grid with white gutters."""
    ch, cw = cell
    grid = np.full((ch * 2 + 12, cw * 2 + 12, 3), (255, 255, 255), np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, 2)
        p = cv2.resize(p, (cw, ch))
        grid[4 + r * (ch + 4): 4 + r * (ch + 4) + ch, 4 + c * (cw + 4): 4 + c * (cw + 4) + cw] = p
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word", default=None, help="show a clip of this word class")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=3, help="same frame sampling as the pipeline")
    ap.add_argument("--save", type=Path, default=None, help="write the visualisation to mp4")
    ap.add_argument("--figure", type=Path, default=None, help="write a single still frame as png")
    ap.add_argument("--invariance", action="store_true",
                    help="show the clip twice under different 'cameras'")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--headless", action="store_true", help="render to files without a window")
    ap.add_argument("--zoom", type=float, default=1.15, help="camera B zoom in --invariance mode")
    ap.add_argument("--shift", type=int, nargs=2, default=(45, 8), help="camera B dx dy")
    a = ap.parse_args()

    clip = pick_clip(a.word, a.video)
    label = clip.parent.name
    print(f"clip: {clip}")

    from .keypoints import LandmarkExtractor
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")

    cell = (360, 480)
    writer = None
    history: list[np.ndarray] = []
    hist_b: list[np.ndarray] = []
    paused, idx, kept = False, 0, 0
    win = "preprocessing"
    if not a.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    ex_a = LandmarkExtractor(running_mode="video")
    ex_b = LandmarkExtractor(running_mode="video") if a.invariance else None

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % a.stride != 0:
                    idx += 1
                    continue
                idx += 1
                kept += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose, left, right, mask = ex_a._frame(rgb, kept * 100)
                feats = build_frame_features(pose[None], left[None], right[None], mask[None])[0]
                history.append(feats)

                shoulder = float(np.linalg.norm(pose[L_SH, :2] - pose[R_SH, :2]))
                shoulder_px = shoulder * frame.shape[1]

                if a.invariance:
                    warped = warp(frame, a.zoom, a.shift[0], a.shift[1])
                    rgb_b = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                    pose_b, left_b, right_b, mask_b = ex_b._frame(rgb_b, kept * 100)
                    hist_b.append(build_frame_features(pose_b[None], left_b[None],
                                                       right_b[None], mask_b[None])[0])
                    sb = float(np.linalg.norm(pose_b[L_SH, :2] - pose_b[R_SH, :2]))
                    panels = [
                        draw_landmarks(frame.copy(), pose, left, right, mask),
                        draw_landmarks(warped.copy(), pose_b, left_b, right_b, mask_b),
                        draw_normalised(cell, pose, left, right, mask, "camera A -> normalised"),
                        draw_normalised(cell, pose_b, left_b, right_b, mask_b, "camera B -> normalised"),
                    ]
                    diff = float(np.abs(history[-1][:27] - hist_b[-1][:27]).mean())
                    caption(panels[0], "camera A: as recorded")
                    caption(panels[1], f"camera B: zoomed {a.zoom:g}x, shifted off-centre")
                    for p in (panels[2], panels[3]):
                        cv2.putText(p, f"mean |A - B| = {diff:.3f}", (14, p.shape[0] - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 120, 40), 2)
                else:
                    stats = [
                        ("word", label, INK),
                        ("frames read / kept", f"{idx} / {kept}  (every {a.stride}rd)", INK),
                        ("shoulder width", f"{shoulder_px:5.1f} px", (230, 90, 40)),
                        ("scale divisor", f"{shoulder:.4f}", (230, 90, 40)),
                        ("hands detected", f"{int(mask.sum())} of 2", TEAL if mask.sum() == 2 else (36, 158, 245)),
                        ("pose points kept", f"{len(POSE_IDX)} of 33", TEAL),
                        ("features this frame", "239", TEAL),
                        ("after velocity", "478", TEAL),
                        ("", "", INK),
                        ("resampled to", "64 frames (fixed grid)", GREY),
                    ]
                    panels = [
                        draw_landmarks(frame.copy(), pose, left, right, mask),
                        draw_normalised(cell, pose, left, right, mask),
                        draw_features(cell, history, feats),
                        draw_stats(cell, stats),
                    ]
                    caption(panels[0], "1. raw frame + what MediaPipe found")

                grid = tile(panels, cell)

                if a.figure and kept == 12:
                    cv2.imwrite(str(a.figure), grid)
                    print(f"wrote {a.figure}")
                if a.save:
                    if writer is None:
                        writer = cv2.VideoWriter(str(a.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 a.fps, (grid.shape[1], grid.shape[0]))
                    writer.write(grid)

            if a.headless:
                continue
            cv2.imshow(win, grid)
            key = cv2.waitKey(int(1000 / a.fps)) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        ex_a.close()
        if ex_b:
            ex_b.close()
        if writer:
            writer.release()
            print(f"wrote {a.save}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
