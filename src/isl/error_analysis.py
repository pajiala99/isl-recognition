"""Error analysis in the regime where the model actually makes mistakes.

On the full split the model is at 100%, so the confusion matrix is uninformative.
We retrain with 2 takes per class and analyse *that* model's errors - those are the
sign pairs the features genuinely struggle to separate.

    python -m isl.error_analysis
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ARTIFACTS, DATA_KP, TrainConfig
from .dataset import ISLSequenceDataset, index_keypoints, stratified_split
from .evaluate import plot_confusion
from .experiments import run_training, subsample_takes
from .train import device_auto


def main(k: int = 2, epochs: int = 40):
    cfg = TrainConfig()
    samples = index_keypoints([DATA_KP])
    classes = sorted({s.label for s in samples})
    train_s, val_s, test_s = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    sub = subsample_takes(train_s, k, cfg.seed)
    model, m = run_training(sub, val_s, test_s, classes, cfg, True, epochs)
    print(f"{k} takes/class -> test top-1 {m['test_top1']*100:.2f}%")

    device = device_auto()
    dl = DataLoader(ISLSequenceDataset(test_s, classes, cfg.seq_len), 64)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    ys, ps = np.array(ys), np.array(ps)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ys, ps, labels=range(len(classes)))
    plot_confusion(cm, classes, ARTIFACTS / "confusion_matrix_k2.png")

    pairs = Counter()
    for t, p in zip(ys, ps):
        if t != p:
            pairs[(classes[t], classes[p])] += 1
    worst_classes = sorted(((classes[i], cm[i, i] / max(cm[i].sum(), 1)) for i in range(len(classes))),
                           key=lambda r: r[1])[:10]
    out = {"takes_per_class": k, "test_top1": m["test_top1"],
           "top_confusions": [{"true": a, "predicted": b, "count": n}
                              for (a, b), n in pairs.most_common(12)],
           "weakest_classes": [{"class": c, "recall": r} for c, r in worst_classes]}
    (ARTIFACTS / "error_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["top_confusions"][:8], indent=2))


if __name__ == "__main__":
    main()
