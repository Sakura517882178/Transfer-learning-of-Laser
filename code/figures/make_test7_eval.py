#!/usr/bin/env python3
"""为每个已训练 run 造 test7 评估用的镜像工作目录 + 任务 json。
镜像结构：train.lmdb = 该 run 的（提供与训练一致的归一化统计）；laser/valid.lmdb = 固定 test7（7 个分子）
"""
import glob, json, os

P = "/mnt/i/green-laser-3core"
W = "/mnt/f/green-laser/3core"
TPL = json.load(open(f"{P}/tasks/test7/template.json"))
TEST7_VALID = f"{P}/data/test7_lmdb/valid.lmdb"

made = 0
for ck in sorted(glob.glob(f"{W}/runs/repeat/tr*_s*/*/save_finetune/checkpoint_best.pt")):
    arm_dir = os.path.dirname(os.path.dirname(ck))
    sid = os.path.basename(os.path.dirname(arm_dir))
    arm = os.path.basename(arm_dir)
    mir = f"{W}/runs/test7eval/{sid}_{arm}"
    os.makedirs(f"{mir}/laser", exist_ok=True)
    links = {f"{mir}/train.lmdb": f"{arm_dir}/train.lmdb",
             f"{mir}/valid.lmdb": f"{arm_dir}/train.lmdb",
             f"{mir}/laser/train.lmdb": f"{arm_dir}/train.lmdb",
             f"{mir}/laser/valid.lmdb": TEST7_VALID,
             f"{mir}/pretrain_dict.txt": f"{arm_dir}/pretrain_dict.txt",
             f"{mir}/laser/pretrain_dict.txt": f"{arm_dir}/pretrain_dict.txt"}
    for dst, src in links.items():
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    cfg = json.load(open(f"{arm_dir}/config.json"))
    json.dump(cfg, open(f"{mir}/config.json", "w"), indent=2)
    json.dump(cfg, open(f"{mir}/laser/config.json", "w"), indent=2)
    t = json.loads(json.dumps(TPL))
    t["id"], t["name"] = f"test7-{sid}-{arm}", f"test7 {sid} {arm}"
    t["inputs"] = {"RUN_DIR": mir, "CKPT": ck, "OUT": f"{W}/report/test7/{sid}_{arm}.csv"}
    json.dump(t, open(f"{P}/tasks/test7/{sid}_{arm}.json", "w"), indent=2, ensure_ascii=False)
    made += 1
print(f"准备 {made} 个 test7 评估任务（镜像目录 + task json）")
print("输出目录:", f"{W}/report/test7/")
