#!/bin/bash
# 验证集排序评估：100 个 run 各自在留出验证集上推理（仅推理，不训练）；resume-safe
set -u
RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
TASKS=/mnt/i/green-laser-3core/tasks/valrank
OUT=/mnt/f/green-laser/3core/report/valrank
LOG="$OUT/valrank.log"
mkdir -p "$OUT"
cd /mnt/f/optoclaw-dev || exit 1
echo "==== 验证集推理开始 $(date '+%F %H:%M:%S') 共 $(ls $TASKS/*.json 2>/dev/null|wc -l) 个 ====" | tee -a "$LOG"
for T in "$TASKS"/*.json; do
  name=$(basename "$T" .json)
  if [ -f "$OUT/$name.csv" ]; then continue; fi
  $PY "$RUNNER" "$T" > /dev/null 2>&1
  if [ -f "$OUT/$name.csv" ]; then echo "OK   $name $(date '+%H:%M:%S')" | tee -a "$LOG"; else echo "FAIL $name" | tee -a "$LOG"; fi
done
echo "==== 完成 $(date '+%F %H:%M:%S') | CSV: $(ls $OUT/*.csv 2>/dev/null|wc -l)/100 ====" | tee -a "$LOG"
