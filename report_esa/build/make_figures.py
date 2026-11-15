"""Figures for the ESA report. Fig 6.1 architecture (drawn), Fig 7.1 data from Table 7.5 (EXP-006/008/009 result files)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#2f4b7c", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arrow(ax, p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))

def fig61(path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.4); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 4.0), 4.4, 6.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#fafafa", ec="#999999", ls="--"))
    ax.add_patch(FancyBboxPatch((5.5, 4.0), 4.4, 6.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#fafafa", ec="#999999", ls="--"))
    ax.text(2.3, 9.9, "Training side (Kaggle, one Tesla T4)", ha="center", fontsize=10, fontweight="bold")
    ax.text(7.7, 9.9, "Analysis side (local computer, CPU)", ha="center", fontsize=10, fontweight="bold")
    box(ax, 0.4, 8.3, 3.8, 1.2, "Inputs: Alpaca data, recipe,\npoison rate, trigger / payload")
    box(ax, 0.4, 6.4, 3.8, 1.2, "M1  Adapter training\nand attack evaluation", bold=True)
    box(ax, 0.4, 4.3, 3.8, 1.3, "M2  Run record\nand SHA-256 checksum", bold=True)
    box(ax, 5.8, 8.3, 3.8, 1.2, "Reference banks: authors' Qwen\nbank, PADBench slices")
    box(ax, 5.8, 6.4, 3.8, 1.2, "M4  Detector fitting\nthreshold chosen, then frozen", bold=True)
    box(ax, 5.8, 4.3, 3.8, 1.3, "M3  Feature extraction\n20 WSD-type features;\nZ-PEFT per-head descriptors", bold=True)
    arrow(ax, (2.3, 8.3), (2.3, 7.6)); arrow(ax, (2.3, 6.4), (2.3, 5.6))
    arrow(ax, (7.7, 8.3), (7.7, 7.6)); arrow(ax, (7.7, 6.4), (7.7, 5.6))
    arrow(ax, (4.2, 4.95), (5.8, 4.95))
    ax.text(5.0, 4.62, "zip", ha="center", fontsize=8, style="italic")
    box(ax, 2.6, 2.2, 4.8, 1.3, "M5  Scoring and readings\nfrozen code hash, gates, score once,\nreading written in advance", bold=True, fc="#fdf3e3", ec="#a86a00")
    arrow(ax, (7.7, 4.3), (6.0, 3.5)); arrow(ax, (2.3, 4.3), (4.0, 3.5))
    box(ax, 1.0, 0.2, 8.0, 1.2, "M6  Tracking: experiment records, run records, adapter manifest, generated index", bold=True, fc="#e8f4ea", ec="#2d6a3e")
    arrow(ax, (5.0, 2.2), (5.0, 1.4))
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

def fig71(path):
    seeds = [0, 1, 2, 3, 4, 5]
    data = {
        "Original recipe": [0.9850, 0.9867, 0.9624, 0.9880, 0.9813, 0.9658],
        "Authors' settings (EXP-008)": [0.9586, 0.9467, 0.9585, 0.9847, 0.9912, 0.9607],
        "Template only": [0.7748, 0.5943, 0.3644, 0.7298, 0.2270, 0.4892],
        "Template + settings": [0.3529, 0.2203, 0.4560, 0.5653, 0.2305, 0.3511],
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for (k, v), m in zip(data.items(), ["o", "s", "^", "D"]):
        ax.plot(seeds, v, marker=m, lw=1.4, label=k)
    ax.axhline(0.4345552465347361, color="k", ls="--", lw=1.2)
    ax.text(5.05, 0.445, "threshold 0.4346", fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("Seed (adapter)"); ax.set_ylabel("Probability assigned 'poisoned' by D_A")
    ax.set_ylim(0, 1.05); ax.set_xticks(seeds); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig61("build/fig_6_1.png"); fig71("build/fig_7_1.png"); print("ok")
