#!/usr/bin/env python3
"""表征对齐迁移（自写一套）：CORAL / OT-Sinkhorn / Procrustes 子空间 / 均值方差匹配
数据：源域与目标域表征均由工作流 extract_emb 从 LMDB 提取（层 5/10/15）
评估：10 个划分 × 5 规模；valid 报数；固定 test 7 全程不参与拟合
对照：① 不对齐  ② 纯目标域(n 标签)  ③ 对齐后源域样本(head)  ④ ③+目标 n 标签重标定  ⑤ 源+目标联合
"""
import csv, glob, json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from scipy.linalg import sqrtm

ROOT = "/mnt/i/green-laser-3core"
REP = "/mnt/f/green-laser/3core/report/align"
SIZES = [6, 13, 27, 40, 53]
TGT = ["S1_eV", "Sigma_opt"]


# ───────── 对齐方法（输入源/目标特征矩阵，返回对齐后的源特征） ─────────
def shrink_cov(X, lam=0.1):
    C = np.cov(X, rowvar=False)
    return C + lam * np.trace(C) / C.shape[0] * np.eye(C.shape[0])


def m_ident(Xs, Xt, Xs_test=None):
    return Xs.copy()


def m_mvmatch(Xs, Xt, Xs_test=None):
    return (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8) * (Xt.std(0) + 1e-8) + Xt.mean(0)


def m_coral(Xs, Xt, Xs_test=None):
    """CORAL：源 → 目标的二阶统计对齐（shrinkage 防秩亏）"""
    Cs, Ct = shrink_cov(Xs), shrink_cov(Xt)
    A = sqrtm(np.linalg.inv(sqrtm(Cs)) @ Ct @ np.linalg.inv(sqrtm(Cs)))  # = Cs^{-1/2} Ct^{1/2}
    return (Xs - Xs.mean(0)) @ A + Xt.mean(0)


def m_procrustes(Xs, Xt, Xs_test=None):
    """子空间对齐（Fernando 2013）：各自 PCA 子空间 → 正交映射把源子空间转到目标子空间"""
    k = max(2, min(20, Xt.shape[0] - 1, Xt.shape[1]))
    Ps = PCA(n_components=k).fit(Xs - Xs.mean(0))
    Pt = PCA(n_components=k).fit(Xt - Xt.mean(0))
    Bs, Bt = Ps.components_.T, Pt.components_.T            # 512×k
    U, _, Vt = np.linalg.svd(Bs.T @ Bt, full_matrices=False)
    R = U @ Vt
    return (Xs - Xs.mean(0)) @ Bs @ R @ Bt.T + Xt.mean(0)


def m_ot(Xs, Xt, Xs_test=None, eps=0.05, iters=200):
    """熵正则最优传输（Sinkhorn）：把源样本搬到目标分布，再做重心映射"""
    Xs_s = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    Xt_s = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)
    C = ((Xs_s[:, None, :] - Xt_s[None, :, :]) ** 2).sum(-1)
    K = np.exp(-C / (eps * C.max()))
    a, b = np.ones(len(Xs_s)) / len(Xs_s), np.ones(len(Xt_s)) / len(Xt_s)
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = (u[:, None] * K) * v[None, :]                    # 耦合矩阵 (n_s × n_t)
    Xs_bar = (P / (P.sum(1, keepdims=True) + 1e-12)) @ Xt_s   # 每个源样本 → 目标点的凸组合
    M, *_ = np.linalg.lstsq(Xs_s, Xs_bar, rcond=None)         # 学到线性搬移
    return ((Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)) @ M * (Xt.std(0) + 1e-8) + Xt.mean(0)


METHODS = {"ident": m_ident, "mv": m_mvmatch, "coral": m_coral, "proc": m_procrustes, "ot": m_ot}


def zlabel(Y, cols=(0, 1)):
    mu, sd = Y.mean(0), Y.std(0) + 1e-8
    return (Y - mu) / sd, mu, sd


def fit_head(X, y, d=None):
    if len(y) < 3:
        return None
    p = PCA(n_components=max(1, min(len(y) - 1, X.shape[1]))).fit(X)
    return RidgeCV(alphas=(0.01, 0.1, 1, 10, 100)).fit(p.transform(X), y), p


def pred_head(model, X):
    m, p = model
    return m.predict(p.transform(X))


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 5     # 用第几层
    zs, z0, z1 = (np.load(f"{REP}/{f}") for f in ("source_0.npz", "target_0.npz", "target_1.npz"))
    sn = [str(x).replace("_train", "").replace("_valid", "") for x in zs["names"]]
    t0n = [str(x).replace("_train", "").replace("_valid", "") for x in z0["names"]]
    t1n = [str(x).replace("_train", "").replace("_valid", "") for x in z1["names"]]
    Xs, Ys = zs[f"L{L}"].astype(float), zs["targets"].astype(float)
    Xtv = np.vstack([z0[f"L{L}"], z1[f"L{L}"]]).astype(float)          # 62 + 7
    Ytv = np.vstack([z0["targets"], z1["targets"]]).astype(float)
    names = t0n + t1n
    idx = {n: i for i, n in enumerate(names)}
    test7 = [r["name"] for r in csv.DictReader(open(f"{ROOT}/splits/test7.csv"))]
    test_i = set([idx[n] for n in test7])
    pool_i = np.array([i for n, i in idx.items() if n not in test_i])
    split_files = sorted(glob.glob(f"{ROOT}/splits/tr*_s*_valid.csv"))
    print(f"源域 {Xs.shape} | 目标域 {Xtv.shape}（test7 {len(test_i)} 固定不参与拟合）| 层 L{L}")

    YS, mu, sd = zlabel(Ys)   # 源标签标准化（解决两域尺度不同）
    rec = {}
    for f in split_files:
        sid = os.path.basename(f).replace("_valid.csv", "")
        n = int(sid.split("_")[0][2:])
        vn = [r["name"] for r in csv.DictReader(open(f))]
        tr_i = np.array([idx[n2] for n2 in names if n2 not in set(vn) and n2 not in set(test7)])
        va_i = np.array([idx[n2] for n2 in vn])
        # 对齐只用：源特征 + 目标(train∪valid)特征；test7 排除
        Xt_fit = Xtv[pool_i]   # 对齐用全部 62 个无标签目标特征（排除 test7），监督只用 n 个 train 标签
        for mname, fn in METHODS.items():
            Xs_al = fn(Xs, Xt_fit)
            pca = PCA(n_components=min(40, len(tr_i) - 1, Xs.shape[1])).fit(np.vstack([Xs_al, Xt_fit]))
            S_al, T_all = pca.transform(Xs_al), pca.transform(Xtv)
            for j, t in enumerate(TGT):
                y_tr, y_va = Ytv[tr_i, j], Ytv[va_i, j]
                ys_std = YS[:, j]
                # ③ 对齐后源域样本训头（零样本）
                h = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100)).fit(S_al, ys_std)
                p3 = h.predict(T_all[va_i]) * sd[j] + mu[j]
                # ④ ③ + 目标 n 标签线性重标定
                p3t = h.predict(T_all[tr_i]) * sd[j] + mu[j]
                A = np.vstack([p3t, np.ones_like(p3t)]).T
                c, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
                p4 = np.vstack([p3, np.ones_like(p3)]).T @ c
                # ⑤ 源(降权) + 目标 n 联合
                Xj = np.vstack([S_al, T_all[tr_i]])
                yj = np.concatenate([ys_std, (y_tr - mu[j]) / sd[j]])
                w = np.concatenate([np.full(len(S_al), 0.1), np.ones(len(tr_i))])
                hj = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100)).fit(Xj, yj, sample_weight=w)
                p5 = hj.predict(T_all[va_i]) * sd[j] + mu[j]
                # ② 纯目标域（对照，PCA 只在目标池上拟合，不含源域）
                pca2 = PCA(n_components=min(40, len(pool_i) - 1, Xtv.shape[1])).fit(Xtv[pool_i])
                T2 = pca2.transform(Xtv)
                h2 = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100)).fit(T2[tr_i], y_tr)
                p2 = h2.predict(T2[va_i])
                rec.setdefault((mname, sid, n, t), {})["2_target"] = (r2_score(y_va, p2), spearmanr(y_va, p2).correlation)
                rec[(mname, sid, n, t)].update({"3_src": (r2_score(y_va, p3), spearmanr(y_va, p3).correlation),
                                                "4_src_cal": (r2_score(y_va, p4), spearmanr(y_va, p4).correlation),
                                                "5_joint": (r2_score(y_va, p5), spearmanr(y_va, p5).correlation)})

    print(f"\n{'方法':<8}{'规模':>5}" + "".join(f"{k:>22}" for k in ("②纯目标", "③源(对齐)", "④源+重标定", "⑤源+目标联合")))
    for mname in METHODS:
        for n in SIZES:
            for t in TGT:
                cells = []
                for k in ("2_target", "3_src", "4_src_cal", "5_joint"):
                    v = [rec[(mname, sid, n, t)][k] for (mm, sid, nn, tt) in rec if mm == mname and nn == n and tt == t]
                    cells.append(f"{np.mean([x[0] for x in v]):>7.3f}/{np.mean([x[1] for x in v]):>5.3f}")
                print(f"{mname:<8}{n:>5}{t:<10}" + "".join(f"{c:>20}" for c in cells))
    json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{kk}": vv for k, d in rec.items() for kk, vv in d.items()},
              open(f"{REP}/align_results_L{L}.json", "w"), indent=2)
    print(f"→ {REP}/align_results_L{L}.json")


if __name__ == "__main__":
    main()
