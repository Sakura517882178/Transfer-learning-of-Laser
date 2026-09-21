#!/bin/bash
# 表征提取（对齐用）：源域 LMDB + 目标域 LMDB → npz（层 5/10/15）
set -u
RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
TASKS=/mnt/i/green-laser-3core/tasks/align
LOG=/mnt/f/green-laser/3core/report/align/extract.log
mkdir -p "$(dirname "$LOG")"
cd /mnt/f/optoclaw-dev || exit 1
echo "==== 表征提取开始 $(date '+%F %H:%M:%S') ====" | tee -a "$LOG"
for T in "$TASKS"/*.json; do
  echo "--- $(basename "$T")" | tee -a "$LOG"
  $PY "$RUNNER" "$T" 2>&1 | grep -E "Saved|\[OK\]|ERROR|层:" | tee -a "$LOG"
done
echo "==== 完成 $(date '+%F %H:%M:%S') ====" | tee -a "$LOG"
ls -la /mnt/f/green-laser/3core/report/align/*.npz 2>/dev/null | awk '{print $5, $9}' | tee -a "$LOG"
