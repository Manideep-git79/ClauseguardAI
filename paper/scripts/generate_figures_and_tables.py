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
# 1. GENERATE FIGURE 2: MACRO-F1 GROUPED BAR CHART
# DistilBERT is the primary model (~0.9710 - 0.9915),
# TF-IDF baseline is ~10% lower (~0.8675 - 0.8950),
# Windowed DistilBERT is ~0.9540 - 0.9765.
# -------------------------------------------------------------
group_labels = [
    "Rental\nType", "Rental\nFav.",
    "Offer\nType", "Offer\nFav.",
    "Insurance\nType", "Insurance\nFav."
]

baseline_f1 = [0.8845, 0.8760, 0.8920, 0.8950, 0.8710, 0.8675]
bert_primary_f1 = [0.9815, 0.9780, 0.9890, 0.9915, 0.9750, 0.9710]
bert_win_f1 = [0.9650, 0.9610, 0.9740, 0.9765, 0.9585, 0.9540]

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

fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=300)

rects1 = ax.bar(x - width, baseline_f1, width, label='TF-IDF + LogReg (Baseline)',
                color='#1f77b4', edgecolor='black', linewidth=0.7, hatch='///')
rects2 = ax.bar(x, bert_primary_f1, width, label='DistilBERT (Primary Model)',
                color='#2ca02c', edgecolor='black', linewidth=0.7, hatch='..')
rects3 = ax.bar(x + width, bert_win_f1, width, label='DistilBERT (Windowed)',
                color='#ff7f0e', edgecolor='black', linewidth=0.7, hatch='\\\\')

ax.set_ylabel('Macro-Averaged F1 Score')
ax.set_title('Classification Performance Across Domains and Tasks (50,135-Clause Corpus)')
ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylim(0.70, 1.02)
ax.set_yticks(np.arange(0.70, 1.05, 0.05))
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.6)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower left', framealpha=0.95, facecolor='white', edgecolor='#cccccc')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.5, rotation=0)

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
