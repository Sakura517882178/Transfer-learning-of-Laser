#!/bin/bash
# 留一模组外推：13 folds × 2 臂 = 26 run；resume-safe
set -u
RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
TASKS=/mnt/i/green-laser-3core/tasks/lomo
TRAIN_ROOT=/mnt/f/green-laser/3core
LOG="$TRAIN_ROOT/report/lomo.log"
mkdir -p "$(dirname "$LOG")"
cd /mnt/f/optoclaw-dev || exit 1
echo "==== LOMO 开始 $(date '+%F %H:%M:%S') 共 $(ls $TASKS/*.json|wc -l) 个 ====" | tee -a "$LOG"
for T in "$TASKS"/*.json; do
  name=$(basename "$T" .json); fold=${name%_*}; arm=${name##*_}
  ck="$TRAIN_ROOT/runs/lomo/$fold/$arm/save_finetune/checkpoint_best.pt"
  if [ -f "$ck" ]; then echo "SKIP  $name" | tee -a "$LOG"; continue; fi
  echo "START $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
  $PY "$RUNNER" "$T" 2>&1 | tee -a "$LOG" | grep -E "AGGREGATE|ERROR" | tail -2
  echo "END   $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
done
echo "==== LOMO 完成 $(date '+%F %H:%M:%S') | $(find $TRAIN_ROOT/runs/lomo -name checkpoint_best.pt|wc -l)/26 ====" | tee -a "$LOG"
