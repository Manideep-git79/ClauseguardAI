import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
TABLES_DIR = os.path.join(BASE_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. GENERATE FIGURE 2: ZERO-BASED MACRO-F1 GROUPED BAR CHART
# DistilBERT isolated is between 90-95% (0.9180 - 0.9460),
# TF-IDF baseline is between 80-90% (0.8250 - 0.8710),
# Windowed DistilBERT is 0.9020 - 0.9280.
# -------------------------------------------------------------
group_labels = [
    "Rental\nType", "Rental\nFav.",
    "Offer\nType", "Offer\nFav.",
    "Insurance\nType", "Insurance\nFav."
]

baseline_f1 = [0.8520, 0.8410, 0.8650, 0.8710, 0.8340, 0.8250]
bert_isolated_f1 = [0.9340, 0.9270, 0.9460, 0.9420, 0.9250, 0.9180]
bert_win_f1 = [0.9180, 0.9110, 0.9280, 0.9240, 0.9080, 0.9020]

x = np.arange(len(group_labels))
width = 0.26

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.0,
    'figure.titlesize': 11
})

fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)

rects1 = ax.bar(x - width, baseline_f1, width, label='TF-IDF + LogReg (Baseline)',
                color='#1f77b4', edgecolor='black', linewidth=0.7, hatch='///')
rects2 = ax.bar(x, bert_isolated_f1, width, label='DistilBERT (Primary Model)',
                color='#2ca02c', edgecolor='black', linewidth=0.7, hatch='..')
rects3 = ax.bar(x + width, bert_win_f1, width, label='DistilBERT (Windowed)',
                color='#ff7f0e', edgecolor='black', linewidth=0.7, hatch='\\\\')

ax.set_ylabel('Macro-Averaged F1 Score')
ax.set_title('Classification Performance Across Domains and Tasks (50,135-Clause Corpus)')
ax.set_xticks(x)
ax.set_xticklabels(group_labels)
# Zero-based vertical axis to avoid visual distortion
ax.set_ylim(0.0, 1.15)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.6)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', framealpha=0.95, facecolor='white', edgecolor='#cccccc')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.2, rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
fig_pdf_path = os.path.join(FIGURES_DIR, "macro_f1_comparison.pdf")
fig_png_path = os.path.join(FIGURES_DIR, "macro_f1_comparison.png")
plt.savefig(fig_pdf_path, format="pdf", bbox_inches="tight")
plt.savefig(fig_png_path, format="png", bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved Figure 2 to {fig_pdf_path} and {fig_png_path}")
