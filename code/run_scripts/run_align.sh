#!/bin/bash
# 表征对齐：对 Uni-Mol 编码器 CLS 表征（层 5/10/15）做 CORAL / OT / 子空间 / 均值方差对齐并评估
set -u
PY=/home/sakura/app/anaconda3/envs/py39-unimol/bin/python3
REP=/mnt/f/green-laser/3core/report/align
SCRIPT=/mnt/i/green-laser-3core/scripts/align_transfer.py
mkdir -p "$REP"
echo "===== 对齐评估开始 $(date '+%F %H:%M:%S') =====" | tee -a "$REP/align_all.log"
for L in 5 10 15; do
  echo "" | tee -a "$REP/align_all.log"
  echo "########## 层 L$L  $(date '+%H:%M:%S') ##########" | tee -a "$REP/align_all.log"
  $PY "$SCRIPT" $L 2>&1 | tee -a "$REP/align_all.log"
done
echo "===== 全部完成 $(date '+%F %H:%M:%S') =====" | tee -a "$REP/align_all.log"
