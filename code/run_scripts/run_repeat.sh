#!/bin/bash
# green-laser-3core 重复实验（10 run：5 规模 × 2 臂，seed 0）——训练根在 F 盘
# 套 templates/run-training-serial.sh 模式：resume-safe（done 跳过、failed 清状态重试）
set -u

RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
TASKS=/mnt/i/green-laser-3core/tasks/repeat      # 规格/清单留项目根（I 盘）
TRAIN_ROOT=/mnt/f/green-laser/3core              # 训练产物落 F 盘
LOG="$TRAIN_ROOT/report/repeat_run.log"
mkdir -p "$(dirname "$LOG")" "$TRAIN_ROOT/runs"
cd /mnt/f/optoclaw-dev || exit 1

task_status() {
  $PY - "$1" <<'EOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
st = [s.get("status") for s in d.get("steps", [])]
print("done" if st and all(x == "done" for x in st) else ("failed" if "failed" in st else ""))
EOF
}

clear_status() {
  $PY - "$1" <<'EOF' 2>/dev/null
import json, sys
p = sys.argv[1]; d = json.load(open(p))
for s in d.get("steps", []):
    s.pop("status", None); s.pop("outputs", None)
json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
EOF
}

echo "==== 重复实验开始 $(date '+%F %H:%M:%S')  训练根=$TRAIN_ROOT ====" | tee -a "$LOG"
for T in "$TASKS"/*.json; do
  name=$(basename "$T")
  st=$(task_status "$T")
  if [ "$st" = "done" ]; then echo "SKIP  $name (done)" | tee -a "$LOG"; continue; fi
  [ "$st" = "failed" ] && { echo "RETRY $name" | tee -a "$LOG"; clear_status "$T"; }
  echo "START $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
  $PY "$RUNNER" "$T" 2>&1 | tee -a "$LOG"
  echo "END   $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
done
echo "==== 重复实验完成 $(date '+%F %H:%M:%S') ====" | tee -a "$LOG"
echo "已完成 run: $(find "$TRAIN_ROOT/runs/repeat" -name checkpoint_best.pt 2>/dev/null | wc -l)/10" | tee -a "$LOG"
