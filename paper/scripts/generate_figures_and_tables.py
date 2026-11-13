import os
import json
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
# -------------------------------------------------------------
group_labels = [
    "Rental\nType", "Rental\nFav.",
    "Offer\nType", "Offer\nFav.",
    "Insurance\nType", "Insurance\nFav."
]

baseline_f1 = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
bert_no_ctx_f1 = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
bert_win_f1 = [0.989, 0.985, 0.992, 0.991, 0.981, 0.977]

x = np.arange(len(group_labels))
width = 0.26

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12
})

fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)

rects1 = ax.bar(x - width, baseline_f1, width, label='TF-IDF + LogReg (Baseline)',
                color='#1f77b4', edgecolor='black', linewidth=0.8, hatch='///')
rects2 = ax.bar(x, bert_no_ctx_f1, width, label='DistilBERT (No Context)',
                color='#ff7f0e', edgecolor='black', linewidth=0.8, hatch='..')
rects3 = ax.bar(x + width, bert_win_f1, width, label='DistilBERT (Windowed)',
                color='#2ca02c', edgecolor='black', linewidth=0.8, hatch='\\\\')

ax.set_ylabel('Macro-Averaged F1 Score')
ax.set_title('Classification Performance Across Domains and Tasks on 50,000-Row Corpus')
ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylim(0, 1.18)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.7)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', framealpha=0.95, facecolor='white', edgecolor='#cccccc')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, rotation=0)

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
