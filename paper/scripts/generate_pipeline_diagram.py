import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8.0
})

fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)

# Card positions: (x, y, w, h)
# col 0: x=2.0
# col 1: x=35.5
# col 2: x=69.0
# top row: y=28.0
# bottom row: y=3.0

boxes = [
    # Top Row: Stage 1 (L), Stage 2 (C), Stage 3 (R)
    (2.0, 28.0, 29.0, 21.0, "1. Ingestion & Adaptive OCR",
     "• PDF, DOCX, TXT Parsing\n• Adaptive OCR Trigger:\n    char < 40, words < 8, alpha < 0.65\n• 300 DPI Fallback Tesseract",
     "#eef6fa", "#1a5276"),
     
    (35.5, 28.0, 29.0, 21.0, "2. PII De-Identification",
     "• Presidio Engine + spaCy NER\n• Redacts Names, SSNs, Phones\n• Retains DATE_TIME for Deadlines\n• Fernet (AES-128-CBC + HMAC)",
     "#eafaf1", "#196f3d"),
     
    (69.0, 28.0, 29.0, 21.0, "3. Clause Segmentation",
     "• Regex Structural Chunking\n• Max Limit: 1,200 chars/clause\n• Sub-chunking: 500 chars/sentence\n• Target Clause Isolation",
     "#fefde8", "#b7950b"),

    # Bottom Row: Stage 6 (L), Stage 5 (C), Stage 4 (R) -> Clean Snake Flow
    (69.0, 3.0, 29.0, 21.0, "4. Multi-Task Classification",
     "• Dual-Task Architecture:\n    - Clause Type (10 classes)\n    - Favorability (3 levels)\n• Calibrated TF-IDF + LogReg\n• Evaluated vs. DistilBERT",
     "#f5eef8", "#6c3483"),
     
    (35.5, 3.0, 29.0, 21.0, "5. Deterministic Risk Engine",
     "• Clause Risk: R_i = w_i · (m_i + Δs) · c_i\n• Missing Detection (τ = 0.08):\n    High: 15, Med: 8, Low: 3\n• Composite: 0.70 R_mean + 0.30 P_miss",
     "#fdedec", "#922b21"),
     
    (2.0, 3.0, 29.0, 21.0, "6. Deadlines & Grounded RAG",
     "• Regex + spaCy Deadline Matcher\n• Prompt Injection Pre-Filter\n• Dense 768-dim Embeddings\n• pgvector Cosine Search (k = 5)\n• Grounded Clause Citations",
     "#ebf5fb", "#21618c")
]

for x, y, w, h, title, body, bg_col, border_col in boxes:
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  facecolor=bg_col, edgecolor=border_col, linewidth=1.1)
    ax.add_patch(rect)
    
    h_bar = patches.FancyBboxPatch((x, y + h - 4.2), w, 4.2, boxstyle="round,pad=0.2,rounding_size=0.6",
                                   facecolor=border_col, edgecolor=border_col, linewidth=0.5)
    ax.add_patch(h_bar)
    
    ax.text(x + w/2.0, y + h - 2.1, title, ha='center', va='center',
            fontsize=7.8, fontweight='bold', color='white')
    
    ax.text(x + 1.2, y + h - 5.5, body, ha='left', va='top',
            fontsize=6.8, color='#1c2833', linespacing=1.28)

# Connectors
arrow_kw = dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.3, color="#2c3e50")

# 1 -> 2 (Right)
ax.annotate("", xy=(35.5, 38.5), xytext=(31.0, 38.5), arrowprops=arrow_kw)
# 2 -> 3 (Right)
ax.annotate("", xy=(69.0, 38.5), xytext=(64.5, 38.5), arrowprops=arrow_kw)

# 3 -> 4 (Down)
ax.annotate("", xy=(83.5, 24.0), xytext=(83.5, 28.0), arrowprops=arrow_kw)

# 4 -> 5 (Left)
ax.annotate("", xy=(64.5, 13.5), xytext=(69.0, 13.5), arrowprops=arrow_kw)
# 5 -> 6 (Left)
ax.annotate("", xy=(31.0, 13.5), xytext=(35.5, 13.5), arrowprops=arrow_kw)

plt.tight_layout()
pdf_path = os.path.join(FIGURES_DIR, "pipeline_architecture.pdf")
png_path = os.path.join(FIGURES_DIR, "pipeline_architecture.png")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
plt.close()
print(f"Re-generated Figure 1 (Snake flow): {pdf_path}")
