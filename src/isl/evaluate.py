"""Stage 3 CLI: test-set metrics, confusion matrix and training-curve plots.

    python -m isl.evaluate --ckpt models/bilstm.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .config import ARTIFACTS, DATA_KP, MODELS, TrainConfig
from .dataset import ISLSequenceDataset, index_keypoints, stratified_split
from .models import build_model
from .train import device_auto

ACCENT = "#0F766E"


def plot_curves(history, path: Path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train", color="#94A3B8")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val", color=ACCENT)
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train", color="#94A3B8")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val top-1", color=ACCENT)
    ax[1].plot(ep, [h["val_top5"] for h in history], label="val top-5", color="#F59E0B", ls="--")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    for a in ax:
        a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_confusion(cm, classes, path: Path):
    n = len(classes)
    cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    side = 7.0 if n > 30 else max(5, n * .55)
    fig, ax = plt.subplots(figsize=(side, side * 0.92))
    im = ax.imshow(cmn, cmap="BuGn", vmin=0, vmax=1)
    if n <= 30:
        ax.set_xticks(range(n), classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), classes, fontsize=8)
    else:
        # 61 class names are illegible at slide size - show the structure instead
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Confusion matrix ({n} classes, row-normalised)", fontsize=11)
    if n <= 25:
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > .55 else "#1F2937")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, fraction=.046)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_comparison(scores: dict, path: Path):
    names = list(scores)
    vals = [scores[k] * 100 for k in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(names, vals, color=["#CBD5E1", "#94A3B8", ACCENT][-len(names):])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylim(0, 105); ax.set_ylabel("test top-1 accuracy (%)")
    ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=MODELS / "bilstm.pt")
    ap.add_argument("--keypoints", type=Path, nargs="+", default=[DATA_KP])
    a = ap.parse_args()

    cfg = TrainConfig()
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    classes = state["classes"]
    samples = index_keypoints(a.keypoints)
    _, _, test_s = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    ds = ISLSequenceDataset(test_s, classes, cfg.seq_len)
    dl = DataLoader(ds, 32)

    device = device_auto()
    model = build_model(state["arch"], len(classes), cfg).to(device)
    model.load_state_dict(state["model"]); model.eval()

    ys, ps, p5 = [], [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(device)).cpu()
            ys += y.tolist(); ps += out.argmax(1).tolist()
            p5 += out.topk(min(5, out.size(1)), 1).indices.tolist()
    ys, ps = np.array(ys), np.array(ps)
    top5 = float(np.mean([y in k for y, k in zip(ys, p5)]))

    report = classification_report(ys, ps, target_names=classes, zero_division=0, digits=3)
    metrics = {"test_top1": float((ys == ps).mean()), "test_top5": top5,
               "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
               "n_test": len(ys), "n_classes": len(classes)}
    print(json.dumps(metrics, indent=2)); print(report)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "classification_report.txt").write_text(report)
    plot_confusion(confusion_matrix(ys, ps, labels=range(len(classes))), classes,
                   ARTIFACTS / "confusion_matrix.png")
    hist_path = ARTIFACTS / f"history_{state['arch']}.json"
    if hist_path.exists():
        plot_curves(json.loads(hist_path.read_text()), ARTIFACTS / "training_curves.png")
    base_path = ARTIFACTS / "baselines.json"
    scores = {}
    if base_path.exists():
        b = json.loads(base_path.read_text())
        scores = {"SVM (RBF)": b["svm_rbf"]["test_top1"],
                  "Random Forest": b["random_forest"]["test_top1"]}
    scores["BiLSTM + Attn"] = metrics["test_top1"]
    plot_comparison(scores, ARTIFACTS / "model_comparison.png")
    print("artifacts written to", ARTIFACTS)


if __name__ == "__main__":
    main()
