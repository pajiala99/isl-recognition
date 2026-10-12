"""Which augmentations actually help on a single-signer dataset, and how noisy is the curve?

    python -m isl.ablation
"""
from __future__ import annotations

import json
import statistics as st

from .config import ARTIFACTS, DATA_KP, TrainConfig
from .dataset import index_keypoints, stratified_split
from .experiments import run_training, subsample_takes

SETTINGS = {
    "none":            None,
    "full":            {},
    "no_mirror":       {"p_mirror": 0.0},
    "geometric_only":  {"p_mirror": 0.0, "speed": 0.0, "p_frame_drop": 0.0},
    "mirror_only":     {"rot_deg": 0.0, "scale": 0.0, "shift": 0.0, "noise": 0.0,
                        "speed": 0.0, "p_frame_drop": 0.0},
}


def main(k: int = 2, seeds=(42, 7), epochs: int = 40):
    cfg = TrainConfig()
    samples = index_keypoints([DATA_KP])
    classes = sorted({s.label for s in samples})
    train_s, val_s, test_s = stratified_split(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    sub = subsample_takes(train_s, k, cfg.seed)

    out = {"takes_per_class": k, "seeds": list(seeds), "settings": {}}
    for name, override in SETTINGS.items():
        scores = []
        for seed in seeds:
            c = TrainConfig()
            if override is not None:
                c.augment = {**c.augment, **override}
            _, m = run_training(sub, val_s, test_s, classes, c, override is not None, epochs, seed)
            scores.append(m["test_top1"])
        out["settings"][name] = {"mean": st.mean(scores), "scores": scores}
        print(f"  {name:<16} {st.mean(scores)*100:6.2f}%   {[round(s*100,1) for s in scores]}", flush=True)

    print("\nseed variance at the full training set (14 takes/class)")
    full_scores = []
    for seed in (42, 7, 13):
        _, m = run_training(subsample_takes(train_s, 14, cfg.seed), val_s, test_s, classes,
                            TrainConfig(), True, epochs, seed)
        full_scores.append(m["test_top1"])
        print(f"  seed {seed}: {m['test_top1']*100:.2f}%", flush=True)
    out["full_training_seeds"] = {"mean": st.mean(full_scores), "scores": full_scores,
                                  "stdev": st.pstdev(full_scores)}

    (ARTIFACTS / "ablation.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", ARTIFACTS / "ablation.json")


if __name__ == "__main__":
    main()
