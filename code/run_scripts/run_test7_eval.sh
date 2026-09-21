#!/bin/bash
# test7 独立评估：遍历 tasks/test7 下所有任务，已有 CSV 则跳过（resume-safe）
set -u
RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
TASKS=/mnt/i/green-laser-3core/tasks/test7
OUT=/mnt/f/green-laser/3core/report/test7
LOG=$OUT/eval.log
mkdir -p "$OUT"; cd /mnt/f/optoclaw-dev || exit 1
echo "==== test7 评估开始 $(date '+%F %H:%M:%S') ====" | tee -a "$LOG"
for T in "$TASKS"/tr*.json; do
  name=$(basename "$T" .json)
  if [ -f "$OUT/$name.csv" ]; then continue; fi
  $PY "$RUNNER" "$T" > /dev/null 2>&1
  if [ -f "$OUT/$name.csv" ]; then echo "OK   $name" | tee -a "$LOG"; else echo "FAIL $name" | tee -a "$LOG"; fi
done
echo "==== 完成 $(date '+%F %H:%M:%S') | CSV: $(ls $OUT/*.csv 2>/dev/null|wc -l) ====" | tee -a "$LOG"
