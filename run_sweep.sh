#!/bin/sh
# Train one model per mixing probability eps and seed (in parallel), then probe them and print the result tables.
# Usage: ./run_sweep.sh [layers] [d_model] [rope|learned]   (default 4 64 rope; set PYTHON to choose the interpreter and SEEDS to
# choose the seeds, by default "0 1 2 3 4")
set -e
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-0 1 2 3 4}
LAYERS=${1:-4}
D_MODEL=${2:-64}
OUT=runs/L${LAYERS}_d${D_MODEL}
case "${3:-rope}" in
  rope) ROPE=--rope; OUT=${OUT}_rope ;;
  learned) ROPE=--no-rope ;;
  *) echo "position embeddings must be rope or learned" >&2; exit 2 ;;
esac
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
