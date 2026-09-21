#!/bin/bash
set -u
cd /mnt/f/optoclaw-dev || exit 1
LOG=/mnt/f/green-laser/3core/report/predict330.log
echo "==== 全库 330 推理 $(date '+%H:%M:%S') ====" | tee -a "$LOG"
/home/sakura/app/anaconda3/bin/python3 engine/workflow_runner.py /mnt/i/green-laser-3core/tasks/predict330_full.json 2>&1 | tee -a "$LOG" | tail -3
ls -la /mnt/f/green-laser/3core/report/predict330_part*.csv 2>/dev/null | awk '{print $5, $NF}' | tee -a "$LOG"
