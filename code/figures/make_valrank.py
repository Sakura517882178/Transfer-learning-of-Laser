#!/usr/bin/env python3
"""为每个已训练 run 造"验证集排序评估"的镜像 + 任务：
  mirror/laser/valid.lmdb → 该 run 自己的 valid.lmdb（留出分子）
  推理输出 → report/valrank/<split>_<arm>.csv
产物：/mnt/i/green-laser-3core/tasks/valrank/*.json（100 个）
"""
import glob
import json
import os

P = "/mnt/i/green-laser-3core"
W = "/mnt/f/green-laser/3core"
os.makedirs(f"{P}/tasks/valrank", exist_ok=True)
os.makedirs(f"{W}/report/valrank", exist_ok=True)

made = 0
for ck in sorted(glob.glob(f"{W}/runs/repeat2/tr*_s*/*/save_finetune/checkpoint_best.pt")):
    arm_dir = os.path.dirname(os.path.dirname(ck))          # .../tr13_s00/transfer
    sid = os.path.basename(os.path.dirname(arm_dir))        # tr13_s00
    arm = os.path.basename(arm_dir)                         # transfer
    mir = f"{W}/runs/valrank/{sid}_{arm}"
    os.makedirs(f"{mir}/laser", exist_ok=True)
    links = {f"{mir}/train.lmdb": f"{arm_dir}/train.lmdb",
             f"{mir}/valid.lmdb": f"{arm_dir}/valid.lmdb",
             f"{mir}/laser/valid.lmdb": f"{arm_dir}/valid.lmdb"}
    for dst, src in links.items():
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    for f in ("config.json", "pretrain_dict.txt", "property_norms.json", "property_norms.pkl"):
        s = f"{arm_dir}/{f}"
        if os.path.exists(s):
            for dst in (f"{mir}/{f}", f"{mir}/laser/{f}"):
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(s), dst)
    if not os.path.exists(f"{arm_dir}/valid.lmdb"):
        print("  ⚠ 无 valid.lmdb:", sid, arm)
    task = {"id": f"valrank-{sid}-{arm}", "name": f"valrank {sid} {arm}",
            "inputs": {"RUN_DIR": mir, "CKPT": ck, "OUT": f"{W}/report/valrank/{sid}_{arm}.csv"},
            "steps": [{"id": "infer", "label": "验证集推理", "tool": "unimol",
                       "params": {"subcommand": "infer_local", "work_dir": mir, "checkpoint": ck,
                                  "split": "valid", "device": "cuda",
                                  "output": f"{W}/report/valrank/{sid}_{arm}.csv"}}]}
    json.dump(task, open(f"{P}/tasks/valrank/{sid}_{arm}.json", "w"), indent=2, ensure_ascii=False)
    made += 1
print(f"准备 {made} 个验证集推理任务")
