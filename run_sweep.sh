#!/bin/sh
# Train one model per mixing probability eps and seed (in parallel), then probe them and write results.md.
# Usage: ./run_sweep.sh [layers] [d_model]   (default 4 64; set PYTHON to choose the interpreter)
set -e
PYTHON=${PYTHON:-python}
LAYERS=${1:-4}
D_MODEL=${2:-64}
OUT=runs/L${LAYERS}_d${D_MODEL}
mkdir -p $OUT
for eps in 0.05 0.01 0; do
  for seed in 0 1 2 3 4; do
    RUN=$OUT/eps${eps}_s${seed}
    $PYTHON train.py --layers $LAYERS --d-model $D_MODEL --eps $eps --seed $seed --out $RUN > $RUN.log 2>&1 &
    PIDS="$PIDS $!"
  done
done
for pid in $PIDS; do wait $pid; done  # stops here if any run failed
$PYTHON probe.py $OUT/*.pt
$PYTHON summarize.py
