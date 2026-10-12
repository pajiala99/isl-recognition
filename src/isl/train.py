"""Stage 2 CLI: train the sequence classifier on cached landmarks.

    python -m isl.train --model bilstm --epochs 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ARTIFACTS, DATA_KP, MODELS, TrainConfig
from .dataset import ISLSequenceDataset, index_keypoints, save_classes, stratified_split
from .models import build_model


def device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def topk_correct(logits, y, k=5):
    k = min(k, logits.size(1))
    return (logits.topk(k, dim=1).indices == y[:, None]).any(1).sum().item()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss = n = c1 = c5 = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += criterion(out, y).item() * len(y)
        c1 += (out.argmax(1) == y).sum().item()
        c5 += topk_correct(out, y)
        n += len(y)
    return loss / max(n, 1), c1 / max(n, 1), c5 / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keypoints", type=Path, nargs="+", default=[DATA_KP])
    ap.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    cfg = TrainConfig()
    if a.epochs: cfg.epochs = a.epochs
    if a.batch_size: cfg.batch_size = a.batch_size
    if a.lr: cfg.lr = a.lr
    tag = a.tag or a.model

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    samples = index_keypoints(a.keypoints)
    if not samples:
        raise SystemExit(f"no .npz under {a.keypoints} - run `python -m isl.extract` first")
    classes = sorted({s.label for s in samples})
    train_s, val_s, test_s = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    print(f"{len(samples)} clips | {len(classes)} classes | "
          f"train {len(train_s)} val {len(val_s)} test {len(test_s)}")

    ds_tr = ISLSequenceDataset(train_s, classes, cfg.seq_len, True, cfg.augment, cfg.seed)
    ds_va = ISLSequenceDataset(val_s, classes, cfg.seq_len)
    ds_te = ISLSequenceDataset(test_s, classes, cfg.seq_len)
    dl_tr = DataLoader(ds_tr, cfg.batch_size, shuffle=True, drop_last=len(ds_tr) > cfg.batch_size)
    dl_va = DataLoader(ds_va, cfg.batch_size)
    dl_te = DataLoader(ds_te, cfg.batch_size)

    device = device_auto()
    model = build_model(a.model, len(classes), cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device {device} | {a.model} | {n_par/1e6:.2f}M params")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * max(1, len(dl_tr)), pct_start=0.25)

    MODELS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS / f"{tag}.pt"
    hist, best, bad, t0 = [], -1.0, 0, time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tl = tn = tc = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); tn += len(y)
        vl, va, va5 = evaluate(model, dl_va, device, criterion)
        hist.append({"epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn,
                     "val_loss": vl, "val_acc": va, "val_top5": va5})
        flag = ""
        if va > best:
            best, bad = va, 0
            torch.save({"model": model.state_dict(), "classes": classes,
                        "arch": a.model, "cfg": cfg.__dict__}, ckpt)
            flag = " *"
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1 or flag:
            print(f"ep {ep:3d} | train {tc/tn:.3f} | val {va:.3f} top5 {va5:.3f}{flag}", flush=True)
        if bad >= cfg.patience:
            print(f"early stop at epoch {ep}")
            break

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    tl, ta, ta5 = evaluate(model, dl_te, device, criterion)
    mins = (time.time() - t0) / 60
    print(f"\nTEST top-1 {ta:.4f} | top-5 {ta5:.4f} | best val {best:.4f} | {mins:.1f} min")

    (ARTIFACTS / f"history_{tag}.json").write_text(json.dumps(hist, indent=2))
    (ARTIFACTS / f"summary_{tag}.json").write_text(json.dumps({
        "arch": a.model, "classes": len(classes), "clips": len(samples),
        "train": len(train_s), "val": len(val_s), "test": len(test_s),
        "params": n_par, "test_top1": ta, "test_top5": ta5, "best_val": best,
        "epochs_run": len(hist), "minutes": mins, "device": str(device),
    }, indent=2))
    save_classes(classes, ARTIFACTS / "classes.json")


if __name__ == "__main__":
    main()
