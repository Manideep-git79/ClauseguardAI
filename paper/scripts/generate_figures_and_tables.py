import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.join(os.path.dirname(BASE_DIR), "clauseguard-ai")
if not os.path.exists(WORKSPACE_ROOT):
    WORKSPACE_ROOT = os.path.dirname(BASE_DIR)

REPORT_PATH = os.path.join(WORKSPACE_ROOT, "ml", "artifacts", "phase3", "phase3_results.json")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
TABLES_DIR = os.path.join(BASE_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

with open(REPORT_PATH, "r") as f:
    report = json.load(f)

# -------------------------------------------------------------
# 1. GENERATE FIGURE 2: MACRO-F1 GROUPED BAR CHART
# -------------------------------------------------------------
domains = [
    ("rental_agreement", "Rental"),
    ("job_offer_letter", "Offer"),
    ("insurance_policy", "Insurance")
]
tasks = [
    ("clause_type", "Type"),
    ("favorability", "Fav.")
]

group_labels = [
    "Rental\nType", "Rental\nFav.",
    "Offer\nType", "Offer\nFav.",
    "Insurance\nType", "Insurance\nFav."
]

baseline_f1 = []
bert_no_ctx_f1 = []
bert_win_f1 = []

for dom_key, _ in domains:
    for task_key, _ in tasks:
        task_data = report[dom_key][task_key]
        baseline_f1.append(task_data["baseline_tfidf_lr"]["f1_macro"])
        bert_no_ctx_f1.append(task_data["bert_no_context"]["f1_macro"])
        bert_win_f1.append(task_data["bert_windowed"]["f1_macro"])

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
ax.set_title('Audited Classification Performance Across Domains and Tasks')
ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylim(0, 1.18)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.7)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#cccccc')

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

# -------------------------------------------------------------
# 2. GENERATE LATEX RESULTS TABLE
# -------------------------------------------------------------
results_tex = r"""\begin{table*}[t]
\centering
\caption{Audited Model Performance Across Document Domains and Evaluation Tasks (Macro-F1, Accuracy, Macro-Precision, and Macro-Recall)}
\label{tab:results}
\small
\setlength{\tabcolsep}{4.6pt}
\begin{tabular}{ll cccc cccc cccc}
\toprule
& & \multicolumn{4}{c}{\textbf{TF-IDF + LogReg (Baseline)}} & \multicolumn{4}{c}{\textbf{DistilBERT (No Context)}} & \multicolumn{4}{c}{\textbf{DistilBERT (Windowed)}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}
\textbf{Domain} & \textbf{Task} & \textbf{F1} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} \\
\midrule
Rental Agreement & Clause Type & \textbf{0.9533} & 0.9521 & 0.9566 & 0.9522 & 0.8566 & 0.8623 & 0.8629 & 0.8630 & 0.5210 & 0.5729 & 0.7025 & 0.5723 \\
 & Favorability & \textbf{0.9338} & 0.9341 & 0.9349 & 0.9328 & 0.7883 & 0.7924 & 0.7983 & 0.7954 & 0.5717 & 0.5768 & 0.6541 & 0.6000 \\
\midrule
Job Offer Letter & Clause Type & \textbf{0.9417} & 0.9440 & 0.9586 & 0.9450 & 0.8550 & 0.8560 & 0.8761 & 0.8571 & 0.8044 & 0.8020 & 0.8211 & 0.8021 \\
 & Favorability & \textbf{0.9643} & 0.9660 & 0.9670 & 0.9636 & 0.8078 & 0.8120 & 0.8080 & 0.8106 & 0.7028 & 0.6920 & 0.7372 & 0.7024 \\
\midrule
Insurance Policy & Clause Type & \textbf{0.8821} & 0.8775 & 0.8836 & 0.8823 & 0.7063 & 0.7095 & 0.7620 & 0.7158 & 0.4085 & 0.4526 & 0.5246 & 0.4334 \\
 & Favorability & \textbf{0.8678} & 0.8597 & 0.8663 & 0.8755 & 0.8094 & 0.8123 & 0.8121 & 0.8117 & 0.6468 & 0.6640 & 0.6538 & 0.6439 \\
\bottomrule
\end{tabular}
\end{table*}
"""

with open(os.path.join(TABLES_DIR, "results_table.tex"), "w", encoding="utf-8") as f:
    f.write(results_tex)
print("Saved tables/results_table.tex")

# -------------------------------------------------------------
# 3. GENERATE LATEX DATASET STATISTICS TABLE
# -------------------------------------------------------------
dataset_tex = r"""\begin{table}[t]
\centering
\caption{Corpus and Dataset Statistics Across Legal Contract Domains (Audited Dataset v2.1-Clean)}
\label{tab:dataset_stats}
\footnotesize
\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{l c c ccc c c}
\toprule
\textbf{Domain} & \textbf{Docs} & \textbf{Clauses} & \textbf{Train} & \textbf{Val} & \textbf{Test} & \textbf{Classes} & $\boldsymbol{\kappa}_{\text{fav}}$ \\
\midrule
Rental Agreement  & 1,932 & 3,334 & 2,332 & 501 & 501 & 10 & 0.8295 \\
Job Offer Letter  & 1,931 & 3,333 & 2,331 & 502 & 500 & 10 & 0.8023 \\
Insurance Policy  & 1,931 & 3,333 & 2,261 & 566 & 506 & 10 & 0.8074 \\
\midrule
\textbf{Total / Overall} & \textbf{5,794} & \textbf{10,000} & \textbf{6,924} & \textbf{1,569} & \textbf{1,507} & \textbf{30} & \textbf{0.8131} \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "dataset_table.tex"), "w", encoding="utf-8") as f:
    f.write(dataset_tex)
print("Saved tables/dataset_table.tex")

# -------------------------------------------------------------
# 4. GENERATE LATEX ABLATION TABLE (Phase 3 through Phase 8)
# -------------------------------------------------------------
ablation_tex = r"""\begin{table}[t]
\centering
\caption{DistilBERT Architectural Optimization and Sequence-Capacity Progression (Clause Type Macro-F1 Across Phases)}
\label{tab:ablation}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l ccc c}
\toprule
\textbf{Experimental Track / Phase} & \textbf{Rental} & \textbf{Offer} & \textbf{Insur.} & \textbf{Mean} \\
\midrule
TF-IDF + Logistic Regression (Baseline) & \textbf{0.9533} & \textbf{0.9417} & \textbf{0.8821} & \textbf{0.9257} \\
Phase 3 DistilBERT Baseline ($\text{seq}=96$) & 0.8566 & 0.8550 & 0.7063 & 0.8060 \\
Phase 3 Naive Concatenation ($\text{seq}=96$) & 0.5210 & 0.8044 & 0.4085 & 0.5780 \\
Phase 5 Target-Preserving Context ($\text{seq}=128$) & 0.8152 & 0.8120 & 0.6840 & 0.7704 \\
Phase 6 Controlled Fine-Tuning ($\text{seq}=128$) & 0.8977 & 0.8651 & 0.7869 & 0.8499 \\
Phase 8 Sequence Capacity Scaled ($\text{seq}=256$) & \textbf{0.9140} & 0.8636 & \textbf{0.8520} & \textbf{0.8765} \\
\bottomrule
\end{tabular}
\end{table}
"""

with open(os.path.join(TABLES_DIR, "ablation_table.tex"), "w", encoding="utf-8") as f:
    f.write(ablation_tex)
print("Saved tables/ablation_table.tex")
