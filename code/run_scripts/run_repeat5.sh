#!/bin/bash
# 3 核域 5 次重复（seed 0–4）× 5 规模 × 2 臂 = 50 run；训练根 F 盘；resume-safe
set -u
RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
TASKS=/mnt/i/green-laser-3core/tasks/repeat5
TRAIN_ROOT=/mnt/f/green-laser/3core
LOG="$TRAIN_ROOT/report/repeat5.log"
mkdir -p "$(dirname "$LOG")"
cd /mnt/f/optoclaw-dev || exit 1
task_status() { $PY - "$1" <<'PYEOF' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1])); st=[s.get("status") for s in d.get("steps",[])]
print("done" if st and all(x=="done" for x in st) else ("failed" if "failed" in st else ""))
PYEOF
}
clear_status() { $PY - "$1" <<'PYEOF' 2>/dev/null
import json,sys
p=sys.argv[1]; d=json.load(open(p))
for s in d.get("steps",[]): s.pop("status",None); s.pop("outputs",None)
json.dump(d,open(p,"w"),indent=2,ensure_ascii=False)
PYEOF
}
echo "==== 5 次重复实验开始 $(date '+%F %H:%M:%S')  共 $(ls $TASKS/*.json|wc -l) 个 ====" | tee -a "$LOG"
for T in "$TASKS"/*.json; do
  name=$(basename "$T")
  ck="$TRAIN_ROOT/runs/repeat/$(echo ${name%.json} | sed 's/_\(transfer\|baseline\)$/\/\1/' | sed 's/^tr/tr/')/save_finetune/checkpoint_best.pt"
  if [ -f "$ck" ]; then echo "SKIP  $name (已有 checkpoint)" | tee -a "$LOG"; continue; fi
  st=$(task_status "$T"); [ "$st" = "failed" ] && clear_status "$T"
  echo "START $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
  $PY "$RUNNER" "$T" 2>&1 | tee -a "$LOG" | grep -E "AGGREGATE|ERROR|Saved" | tail -3
  echo "END   $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
done
echo "==== 完成 $(date '+%F %H:%M:%S') | 已有 checkpoint: $(find $TRAIN_ROOT/runs/repeat -name checkpoint_best.pt|wc -l)/50 ====" | tee -a "$LOG"
