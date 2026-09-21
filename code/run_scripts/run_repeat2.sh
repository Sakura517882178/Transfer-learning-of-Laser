#!/bin/bash
# 3 核域重训（σ 口径修正后）：seed 0–9 × 5 规模 × 2 臂 = 100 run；resume-safe
set -u
RUNNER=/mnt/f/optoclaw-dev/engine/workflow_runner.py
PY=/home/sakura/app/anaconda3/bin/python3
TASKS=/mnt/i/green-laser-3core/tasks/repeat2
TRAIN_ROOT=/mnt/f/green-laser/3core
LOG="$TRAIN_ROOT/report/repeat2.log"
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
echo "==== 重训（σ×π 修正标签）开始 $(date '+%F %H:%M:%S')  共 $(ls $TASKS/*.json 2>/dev/null|wc -l) 个 ====" | tee -a "$LOG"
for T in "$TASKS"/*.json; do
  name=$(basename "$T")
  base="${name%.json}"; split="${base%_*}"; arm="${base##*_}"
  ck="$TRAIN_ROOT/runs/repeat2/$split/$arm/save_finetune/checkpoint_best.pt"
  if [ -f "$ck" ]; then echo "SKIP  $name (已有 checkpoint)" | tee -a "$LOG"; continue; fi
  st=$(task_status "$T"); [ "$st" = "failed" ] && clear_status "$T"
  echo "START $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
  $PY "$RUNNER" "$T" 2>&1 | tee -a "$LOG" | grep -E "AGGREGATE|ERROR|Saved" | tail -3
  echo "END   $name  $(date '+%H:%M:%S')" | tee -a "$LOG"
done
echo "==== 完成 $(date '+%F %H:%M:%S') | checkpoint: $(find $TRAIN_ROOT/runs/repeat2 -name checkpoint_best.pt 2>/dev/null|wc -l)/100 ====" | tee -a "$LOG"
