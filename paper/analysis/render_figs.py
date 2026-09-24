#!/usr/bin/env python3
"""Render the interpretability figures from interp.npz (no GPU needed)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "interp.npz", allow_pickle=True)
y = d["y"]; classes = [str(c) for c in d["classes"]]
CH = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8",
      "T3","T4","T5","T6","A1","A2","FZ","CZ","PZ","ROC"]
COLORS = {"artf": "#d1495b", "bckg": "#8d96a3", "epileptiform": "#2e7d32", "eyem": "#edae49"}
LAB = {"artf": "Artifact", "bckg": "Background", "epileptiform": "Epileptiform", "eyem": "Eye movement"}

# ---------------- Figure: embedding t-SNE (Bonn + Siena) ----------------
have_bonn = "bonn_tsne" in d.files
have_siena = "siena_tsne_frozen" in d.files
ncol = (1 if have_bonn else 0) + (2 if have_siena else 0)
fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.4))
if ncol == 1:
    axes = [axes]
col = 0
tag_letters = ["(a)", "(b)", "(c)", "(d)"]

# (a) Bonn seizure vs normal (frozen) -- clean 2-class case
if have_bonn:
    axb = axes[col]
    Zb = d["bonn_tsne"]; yb = d["bonn_y"]
    bcol = {0: "#8d96a3", 1: "#c1121f"}; blab = {0: "Normal", 1: "Seizure"}
    kb = float(d["knn_bonn_frozen"]) if "knn_bonn_frozen" in d.files else None
    for ci in [0, 1]:
        m = yb == ci
        axb.scatter(Zb[m, 0], Zb[m, 1], s=9, alpha=0.75, c=bcol[ci], label=blab[ci], linewidths=0)
    ttl = f"{tag_letters[col]} Bonn: seizure vs. normal"
    if kb is not None:
        ttl += f"\n(frozen; bal. kNN {kb:.2f})"
    axb.set_title(ttl, fontsize=9)
    axb.legend(loc="best", fontsize=7, framealpha=0.9, markerscale=1.5)
    axb.set_xticks([]); axb.set_yticks([])
    for s in axb.spines.values():
        s.set_edgecolor("#bbbbbb")
    col += 1

# (b,c) Siena seizure vs normal, frozen -> finetuned+RL: the clearest
# frozen-to-finetuned improvement among the datasets we analyzed.
if have_siena:
    sy = d["siena_y"]
    scol = {0: "#8d96a3", 1: "#c1121f"}; slab = {0: "Normal", 1: "Seizure"}
    knn_fz = float(d["knn_siena_frozen"]) if "knn_siena_frozen" in d.files else None
    knn_ft = float(d["knn_siena_finetuned"]) if "knn_siena_finetuned" in d.files else None
    panels = [("siena_tsne_frozen", "Siena: frozen embedding", knn_fz),
              ("siena_tsne_finetuned", "Siena: finetuned + RL embedding", knn_ft)]
    for key, title, kk in panels:
        ax = axes[col]
        Z = d[key]
        for ci in [0, 1]:
            m = sy == ci
            ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.75, c=scol[ci], label=slab[ci], linewidths=0)
        t = f"{tag_letters[col]} {title}"
        if kk is not None:
            t += f"\n(bal. kNN {kk:.2f})"
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbbbbb")
        if col == ncol - 1:
            ax.legend(loc="best", fontsize=7, framealpha=0.9, markerscale=1.5)
        col += 1

plt.tight_layout()
plt.savefig(HERE / "fig_tsne.pdf", bbox_inches="tight", dpi=200)
plt.savefig(HERE / "fig_tsne.png", bbox_inches="tight", dpi=160)
plt.close()
print("wrote fig_tsne")

# ---------------- Figure 2: saliency (topomap + channel bar + time) ----------------
sal_ch = d["sal_channel"]            # (22,)
sal_time = d["sal_time"]             # (1280,)
sc = (sal_ch - sal_ch.min()) / (sal_ch.max() - sal_ch.min() + 1e-9)

fig = plt.figure(figsize=(7.2, 3.0))
ax0 = fig.add_subplot(1, 3, 1)
topo_ok = False
try:
    import mne
    keep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,20]  # drop A1,A2,ROC
    mne_names = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8",
                 "T7","T8","P7","P8","Fz","Cz","Pz"]
    info = mne.create_info(mne_names, sfreq=256.0, ch_types="eeg")
    info.set_montage("standard_1020")
    vals = sc[keep]
    im, _ = mne.viz.plot_topomap(vals, info, axes=ax0, show=False, cmap="YlOrRd",
                                 contours=4, sensors=True)
    ax0.set_title("(a) Scalp saliency", fontsize=9)
    topo_ok = True
except Exception as e:
    print("topomap failed, fallback bar:", e)
if not topo_ok:
    ax0.axis("off")

ax1 = fig.add_subplot(1, 3, 2)
order = np.argsort(sal_ch)
ax1.barh(range(22), sal_ch[order], color="#2e7d32", height=0.8)
ax1.set_yticks(range(22)); ax1.set_yticklabels([CH[i] for i in order], fontsize=5.5)
ax1.set_xlabel("mean |saliency|", fontsize=8)
ax1.set_title("(b) Per-channel importance", fontsize=9)
ax1.tick_params(axis="x", labelsize=6)
for s in ["top", "right"]:
    ax1.spines[s].set_visible(False)

ax2 = fig.add_subplot(1, 3, 3)
t = np.linspace(0, 5, len(sal_time))
ax2.plot(t, sal_time, color="#2e7d32", lw=1.2)
ax2.fill_between(t, sal_time, color="#2e7d32", alpha=0.15)
ax2.set_xlabel("time (s)", fontsize=8); ax2.set_ylabel("mean |saliency|", fontsize=8)
ax2.set_title("(c) Temporal profile", fontsize=9)
ax2.tick_params(labelsize=6)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig(HERE / "fig_saliency.pdf", bbox_inches="tight", dpi=200)
plt.savefig(HERE / "fig_saliency.png", bbox_inches="tight", dpi=160)
plt.close()
print("wrote fig_saliency")

from sklearn.metrics import silhouette_score
for key, name in [("tsne_frozen", "frozen"), ("tsne_finetuned", "finetuned")]:
    try:
        s = silhouette_score(d[key], y)
        print(f"silhouette ({name} t-SNE): {s:.3f}")
    except Exception as e:
        print("silhouette failed", e)
