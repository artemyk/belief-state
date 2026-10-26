#!/bin/sh
# Train one model per mixing probability eps and seed (in parallel), then probe them and write results.md.
# Usage: ./run_sweep.sh [layers] [d_model] [rope]   (default 4 64; set PYTHON to choose the interpreter and SEEDS to
# choose the seeds, by default "0 1 2 3 4")
set -e
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-0 1 2 3 4}
LAYERS=${1:-4}
D_MODEL=${2:-64}
OUT=runs/L${LAYERS}_d${D_MODEL}
if [ "$3" = rope ]; then
  ROPE=--rope
  OUT=${OUT}_rope
fi
for eps in 0.05 0; do
  mkdir -p $OUT/eps${eps}
  for seed in $SEEDS; do
    RUN=$OUT/eps${eps}/s${seed}
    $PYTHON train.py --layers $LAYERS --d-model $D_MODEL $ROPE --eps $eps --seed $seed --out $RUN > $RUN.log 2>&1 &
    PIDS="$PIDS $!"
  done
done
for pid in $PIDS; do wait $pid; done  # stops here if any run failed
$PYTHON probe.py $OUT/*/*.pt
$PYTHON summarize.py
