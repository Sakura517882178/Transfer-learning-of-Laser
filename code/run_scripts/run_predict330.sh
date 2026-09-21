#!/bin/bash
set -u
RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
LOG=/mnt/f/green-laser/3core/report/predict330.log
mkdir -p "$(dirname "$LOG")"
cd /mnt/f/optoclaw-dev || exit 1
echo "==== 330 候选预测开始 $(date '+%F %H:%M:%S') ====" | tee -a "$LOG"
$PY "$RUNNER" /mnt/i/green-laser-3core/tasks/predict330.json 2>&1 | tee -a "$LOG" | tail -5
echo "==== 完成 $(date '+%F %H:%M:%S') ====" | tee -a "$LOG"
ls -la /mnt/f/green-laser/3core/report/predict330_*.csv 2>/dev/null | awk '{print $5, $NF}' | tee -a "$LOG"
