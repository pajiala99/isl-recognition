#!/usr/bin/env bash
# End-to-end pipeline. Run from the project root.
set -euo pipefail
export PYTHONPATH="$PWD/src"
python -m isl.extract   --videos data/videos --out data/keypoints --workers 5
python -m isl.baselines
python -m isl.train     --model bilstm
python -m isl.evaluate  --ckpt models/bilstm.pt
