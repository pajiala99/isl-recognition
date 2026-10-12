"""Stage 4 CLI: the experiments that matter once the standard split saturates.

The published split of this dataset is degenerate - every class was recorded in a
single five-minute session by one signer, so 20 takes of a word are near-copies and
even a Random Forest on time-pooled features scores 100%. These protocols measure
what the saturated benchmark cannot:

  A  data efficiency   - accuracy vs number of takes per class (k-shot)
  B  robustness        - accuracy under camera / tempo / occlusion perturbations
  C  augmentation      - same model trained with and without augmentation

    python -m isl.experiments --epochs 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ARTIFACTS, DATA_KP, TrainConfig
from .dataset import (ISLSequenceDataset, _mirror, _rot_scale_shift, index_keypoints,
                      stratified_split)
from .features import resample, with_deltas
from .models import build_model
from .train import device_auto, evaluate


def subsample_takes(train, k, seed):
    """Keep at most k recording groups (takes) per class."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, dict[str, list]] = {}
    for s in train:
        by_class.setdefault(s.label, {}).setdefault(s.group or str(s.path), []).append(s)
    out = []
    for label, groups in sorted(by_class.items()):
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys[:k]:
            out += groups[key]
    return out


def run_training(train_s, val_s, test_s, classes, cfg, augment=True, epochs=40, seed=42):
    torch.manual_seed(seed)
    ds_tr = ISLSequenceDataset(train_s, classes, cfg.seq_len, augment,
                               cfg.augment if augment else None, seed)
    dl_tr = DataLoader(ds_tr, cfg.batch_size, shuffle=True)
    dl_va = DataLoader(ISLSequenceDataset(val_s, classes, cfg.seq_len), cfg.batch_size)
    dl_te = DataLoader(ISLSequenceDataset(test_s, classes, cfg.seq_len), cfg.batch_size)

    device = device_auto()
    model = build_model("bilstm", len(classes), cfg).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr,
                                                total_steps=epochs * max(1, len(dl_tr)), pct_start=0.25)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        _, va, _ = evaluate(model, dl_va, device, crit)
        if va > best:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    model.load_state_dict(best_state)
    _, ta, ta5 = evaluate(model, dl_te, device, crit)
    return model, {"val": best, "test_top1": ta, "test_top5": ta5}


# ------------------------------------------------------------------ perturbations
def perturb(seq, kind, rng):
    if kind == "clean":
        return seq
    if kind == "mirror":
        return _mirror(seq)
    if kind == "rotate_15":
        return _rot_scale_shift(seq, 15.0, 0.0, 0.0, rng)
    if kind == "scale_20":
        return _rot_scale_shift(seq, 0.0, 0.2, 0.0, rng)
    if kind == "shift_10":
        return _rot_scale_shift(seq, 0.0, 0.0, 0.10, rng)
    if kind == "speed_50":
        return resample(seq, max(8, int(len(seq) * rng.choice([0.5, 1.5]))))
    if kind == "frame_drop_30":
        keep = rng.random(len(seq)) > 0.3
        return seq[keep] if keep.sum() >= 8 else seq
    if kind == "noise_03":
        return seq + rng.normal(0, 0.03, seq.shape).astype(np.float32)
    if kind == "occlude_left_hand":
        out = seq.copy()
        out[:, 27:69] = 0.0          # body-relative left hand
        out[:, 111:174] = 0.0        # wrist-local left hand
        out[:, -2] = 0.0             # presence flag
        return out
    raise ValueError(kind)


@torch.no_grad()
def robustness(model, test_s, classes, cfg, seed=0):
    device = device_auto()
    model.eval()
    idx = {c: i for i, c in enumerate(classes)}
    base = ISLSequenceDataset(test_s, classes, cfg.seq_len)
    raw = [base.raw(i) for i in range(len(base))]
    y = torch.tensor([idx[s.label] for s in test_s])
    out = {}
    for kind in ["clean", "mirror", "rotate_15", "scale_20", "shift_10", "speed_50",
                 "frame_drop_30", "noise_03", "occlude_left_hand"]:
        rng = np.random.default_rng(seed)
        X = torch.stack([torch.from_numpy(with_deltas(resample(perturb(s, kind, rng), cfg.seq_len)))
                         for s in raw])
        preds = []
        for i in range(0, len(X), 64):
            preds.append(model(X[i:i + 64].to(device)).argmax(1).cpu())
        out[kind] = float((torch.cat(preds) == y).float().mean())
        print(f"  {kind:<18} {out[kind]*100:6.2f}%", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keypoints", type=Path, nargs="+", default=[DATA_KP])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3, 5, 10, 14])
    a = ap.parse_args()

    cfg = TrainConfig()
    samples = index_keypoints(a.keypoints)
    classes = sorted({s.label for s in samples})
    train_s, val_s, test_s = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    print(f"{len(samples)} clips | {len(classes)} classes | train {len(train_s)} "
          f"val {len(val_s)} test {len(test_s)}\n")

    results = {"n_classes": len(classes), "n_clips": len(samples)}

    print("A. data efficiency (takes per class)")
    curve = []
    full_model = None
    for k in a.shots:
        sub = subsample_takes(train_s, k, cfg.seed)
        model, m = run_training(sub, val_s, test_s, classes, cfg, True, a.epochs)
        curve.append({"takes_per_class": k, "train_clips": len(sub), **m})
        print(f"  k={k:<3} clips={len(sub):<4} test top-1 {m['test_top1']*100:6.2f}%  "
              f"top-5 {m['test_top5']*100:6.2f}%", flush=True)
        full_model = model
    results["data_efficiency"] = curve

    print("\nB. robustness of the full model")
    results["robustness"] = robustness(full_model, test_s, classes, cfg)

    print("\nC. augmentation ablation (2 takes per class)")
    sub = subsample_takes(train_s, 2, cfg.seed)
    _, with_aug = run_training(sub, val_s, test_s, classes, cfg, True, a.epochs)
    _, no_aug = run_training(sub, val_s, test_s, classes, cfg, False, a.epochs)
    results["augmentation_ablation"] = {"with_augmentation": with_aug, "without_augmentation": no_aug}
    print(f"  with augmentation    {with_aug['test_top1']*100:6.2f}%")
    print(f"  without augmentation {no_aug['test_top1']*100:6.2f}%")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "experiments.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", ARTIFACTS / "experiments.json")


if __name__ == "__main__":
    main()
