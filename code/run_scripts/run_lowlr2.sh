#!/bin/bash
# 低学习率微调（lr=1e-5，抑制灾难性遗忘）× 5 规模 × 10 划分 = 50 run；resume-safe
set -u
RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
TASKS=/mnt/i/green-laser-3core/tasks/lowlr2
TRAIN_ROOT=/mnt/f/green-laser/3core
LOG="$TRAIN_ROOT/report/lowlr2.log"
mkdir -p "$(dirname "$LOG")"
cd /mnt/f/optoclaw-dev || exit 1
echo "==== 低学习率微调（lr=1e-5）开始 $(date '+%F %H:%M:%S')  共 $(ls $TASKS/*.json 2>/dev/null|wc -l) 个 ====" | tee -a "$LOG"
for T in "$TASKS"/*.json; do
  name=$(basename "$T" .json)
  ck="$TRAIN_ROOT/runs/lowlr2/$(echo $name | sed 's/_transfer$//')/transfer/save_finetune/checkpoint_best.pt"
  if [ -f "$ck" ]; then echo "SKIP  $name" | tee -a "$LOG"; continue; fi
  echo "START $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
  $PY "$RUNNER" "$T" 2>&1 | tee -a "$LOG" | grep -E "AGGREGATE|ERROR" | tail -2
  echo "END   $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
done
echo "==== 完成 $(date '+%F %H:%M:%S') | checkpoint: $(find $TRAIN_ROOT/runs/lowlr2 -name checkpoint_best.pt 2>/dev/null|wc -l)/50 ====" | tee -a "$LOG"
