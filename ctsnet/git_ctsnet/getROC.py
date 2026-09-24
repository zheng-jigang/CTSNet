import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

SCORE_DIR = r"D:\PythonProject3\ctsnet\work\score"
OUT_DIR = r"D:\PythonProject3\ctsnet\work\ROC\ROC_6model"
os.makedirs(OUT_DIR, exist_ok=True)

source_list = ["CASIA", "COEP", "Tongji", "IITD"]
target_list = ["CASIA", "COEP", "Tongji", "IITD"]
model_names = ["ccnet", "compnet", "ctsnet", "sf2net","resnet18",'vgg16']

def load_score_txt(path):
    sims = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, l = line.split()
            sims.append(float(s))
            labels.append(int(l))
    return np.array(sims, dtype=np.float64), np.array(labels, dtype=np.int32)

def sim_label_to_far_frr_fast(sims, labels):
    genuine = sims[labels == 1]
    impostor = sims[labels == -1]
    if len(genuine) == 0 or len(impostor) == 0:
        return np.array([]), np.array([])

    g_sorted = np.sort(genuine)
    i_sorted = np.sort(impostor)
    all_thresh = np.unique(sims)
    all_thresh = np.sort(all_thresh)[::-1]

    frr = np.searchsorted(g_sorted, all_thresh, side="right") / len(g_sorted)
    far = (len(i_sorted) - np.searchsorted(i_sorted, all_thresh, side="right")) / len(i_sorted)
    return far, frr

def compute_eer(far, frr):
    if len(far) == 0 or len(frr) == 0:
        return np.nan
    diff = far - frr
    idx = np.argwhere(np.diff(np.sign(diff))).flatten()
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    x1, y1 = far[i], frr[i]
    x2, y2 = far[i+1], frr[i+1]
    if (x2 - x1) - (y2 - y1) == 0:
        return (x1 + y1) / 2
    t = (y1 - x1) / ((x2 - x1) - (y2 - y1))
    eer = x1 + t * (x2 - x1)
    return eer

def smooth_roc_curve(far, frr, num_points=500):
    """对数空间插值平滑ROC曲线"""
    tpr = 1.0 - frr
    # 过滤重复/异常点
    mask = (far > 0) & np.isfinite(far) & np.isfinite(tpr)
    far = far[mask]
    tpr = tpr[mask]
    if len(far) < 3:
        return far, tpr
    # 对数均匀采样
    log_far = np.log10(far)
    log_far_new = np.linspace(log_far.min(), log_far.max(), num_points)
    far_new = 10 ** log_far_new
    interp_func = interp1d(log_far, tpr, kind="linear", fill_value="extrapolate")
    tpr_new = interp_func(log_far_new)
    tpr_new = np.clip(tpr_new, 0.0, 1.0)
    return far_new, tpr_new

def plot_roc_for_target(target_ds, source_ds, model_curves, save_path, use_log_x=True):
    plt.figure(figsize=(6,5))
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    linestyles = ["-","--","-.",":"]
    for idx, (mname, (far, frr)) in enumerate(model_curves.items()):
        if len(far) == 0:
            continue
        far_smooth, tpr_smooth = smooth_roc_curve(far, frr)
        plt.plot(far_smooth, tpr_smooth, label=mname, color=colors[idx%4],
                 linestyle=linestyles[idx%4], linewidth=1.4)
    plt.plot([0,1],[0,1],"k--", lw=1.0, alpha=0.6)
    if use_log_x:
        plt.xscale("log")
        plt.xlim(1e-4, 1)
    else:
        plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("FPR (FAR)")
    plt.ylabel("TPR (1-FRR)")
    plt.title(f"Source:{source_ds} → Target:{target_ds}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

if __name__ == "__main__":
    skip_same_domain = False
    for src in source_list:
        for tgt in target_list:
            if skip_same_domain and src == tgt:
                print(f"Skip same domain {src}→{tgt}")
                continue
            curve_dict = {}
            eer_dict = {}
            for m in model_names:
                fname = f"{src}_{tgt}_{m}_score.txt"
                fpath = os.path.join(SCORE_DIR, fname)
                if not os.path.exists(fpath):
                    print(f"Skip missing: {fpath}")
                    continue
                sims, labels = load_score_txt(fpath)
                far, frr = sim_label_to_far_frr_fast(sims, labels)
                curve_dict[m] = (far, frr)
                eer_dict[m] = compute_eer(far, frr)
            if len(curve_dict) == 0:
                continue
            out_pdf = os.path.join(OUT_DIR, f"ROC_{src}_to_{tgt}.pdf")
            plot_roc_for_target(tgt, src, curve_dict, out_pdf, use_log_x=True)
            print(f"Saved: {out_pdf}")
            print(f"EER {src}→{tgt}: {eer_dict}")