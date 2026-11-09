import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for IEEE publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 10

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path("results/metrics")

print("Generating Figure 1: Pipeline Diagram...")
# -------------------------------------------------------------
# Figure 1: Pipeline Diagram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.1, 2.6), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis('off')

def draw_box(x, y, w, h, text, bg="#f0f4f8", border="#1f497d", title=""):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                  facecolor=bg, edgecolor=border, linewidth=1.2)
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 2.5, title, ha='center', va='center',
                fontsize=8, fontweight='bold', color=border)
        ax.text(x + w/2, y + (h-2.5)/2, text, ha='center', va='center',
                fontsize=7.5, color='#111111')
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=8, color='#111111')

def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 1.2, label, ha='center', va='center',
                fontsize=7, color="#333333", style='italic')

# Boxes
draw_box(1, 14, 15, 12, "175,341 Train\n82,332 Test\n45 Columns", bg="#eef2f7", border="#2c3e50", title="UNSW-NB15")
draw_arrow(16.5, 20, 21.5, 20)

draw_box(21.5, 14, 17, 12, "Drop id & attack_cat\nOne-Hot Categoricals\n194 Encoded Feats", bg="#eef2f7", border="#2c3e50", title="Preprocessing")
draw_arrow(39, 20, 44, 30)
draw_arrow(39, 20, 44, 20)
draw_arrow(39, 20, 44, 10)

# 3 Branches
draw_box(44, 25, 22, 10, "Baseline (All)\n194 Features (0% cut)", bg="#f8f9fa", border="#495057")
draw_box(44, 15, 22, 10, "Mutual Information\nTop 97 Features (50% cut)", bg="#e8f4f8", border="#17a2b8")
draw_box(44, 5, 22, 10, "RF Gini Importance\nTop 97 Features (50% cut)", bg="#eaf2e8", border="#28a745")

draw_arrow(66.5, 30, 71.5, 20)
draw_arrow(66.5, 20, 71.5, 20)
draw_arrow(66.5, 10, 71.5, 20)

draw_box(71.5, 14, 12, 12, "200 Trees\nmax_features='sqrt'\nrandom_state=42", bg="#fff9e6", border="#e0a800", title="Random Forest")
draw_arrow(84, 20, 88, 20)

draw_box(88, 12, 11, 16, "Accuracy, F1\nInference Time\nSeen vs Unseen\nConflict Audit", bg="#fbeee6", border="#d9534f", title="Evaluation")

plt.tight_layout()
fig.savefig(OUT_DIR / "pipeline.pdf", bbox_inches='tight')
fig.savefig(OUT_DIR / "pipeline.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("Generating Figure 2: Performance Comparison Bar Chart...")
# -------------------------------------------------------------
# Figure 2: Performance Comparison Bar Chart
# -------------------------------------------------------------
# Load results
df_base = pd.read_csv(METRICS_DIR / "baseline_results.csv")
df_mi = pd.read_csv(METRICS_DIR / "mutual_information_results.csv")
df_rf = pd.read_csv(METRICS_DIR / "random_forest_importance_results.csv")

metrics = ['Accuracy', 'Attack_F1', 'Macro_F1', 'False_Positive_Rate']
metric_labels = ['Accuracy', 'Attack F1', 'Macro-F1', 'FPR (Lower is better)']

baseline_vals = [df_base['Accuracy'].iloc[0]*100, df_base['Attack_F1'].iloc[0]*100,
                 df_base['Macro_F1'].iloc[0]*100, df_base['False_Positive_Rate'].iloc[0]*100]

mi_vals = [df_mi['Accuracy'].iloc[0]*100, df_mi['Attack_F1'].iloc[0]*100,
           df_mi['Macro_F1'].iloc[0]*100, df_mi['False_Positive_Rate'].iloc[0]*100]

rf_vals = [df_rf['Accuracy'].iloc[0]*100, df_rf['Attack_F1'].iloc[0]*100,
           df_rf['Macro_F1'].iloc[0]*100, df_rf['False_Positive_Rate'].iloc[0]*100]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)

rects1 = ax.bar(x - width, baseline_vals, width, label='Baseline (194 feats)', color='#4a6984', edgecolor='black', linewidth=0.7)
rects2 = ax.bar(x, mi_vals, width, label='Mutual Information (97 feats)', color='#5fa8a3', edgecolor='black', linewidth=0.7)
rects3 = ax.bar(x + width, rf_vals, width, label='RF Importance (97 feats)', color='#e07a5f', edgecolor='black', linewidth=0.7)

ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontweight='medium')
ax.set_ylim(0, 115)
ax.legend(loc='upper right', frameon=True, edgecolor='#cccccc')

# Add direct value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR / "comparison_bar.pdf", bbox_inches='tight')
fig.savefig(OUT_DIR / "comparison_bar.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("Generating Figure 3: Top-20 Feature Importance (MI & RF)...")
# -------------------------------------------------------------
# Figure 3: Top-20 Feature Importance Side-by-Side
# -------------------------------------------------------------
mi_ranking = pd.read_csv(METRICS_DIR / "mutual_information_feature_ranking.csv").head(20)
rf_ranking = pd.read_csv(METRICS_DIR / "random_forest_feature_ranking.csv").head(20)

# Clean up feature names for clean plotting
rf_ranking['Clean_Feature'] = rf_ranking['Feature'].str.replace('numerical__', '').str.replace('categorical__', '')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.5), dpi=300)

# MI plot
y_pos = np.arange(len(mi_ranking))
ax1.barh(y_pos, mi_ranking['MI_Score'][::-1], color='#5fa8a3', edgecolor='black', linewidth=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(mi_ranking['Feature'][::-1], fontsize=7.5)
ax1.set_xlabel('Mutual Information Score')
ax1.set_title('(a) Top 20 by Mutual Information', fontsize=9, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# RF plot
ax2.barh(y_pos, rf_ranking['Importance'][::-1], color='#e07a5f', edgecolor='black', linewidth=0.6)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(rf_ranking['Clean_Feature'][::-1], fontsize=7.5)
ax2.set_xlabel('Mean Decrease in Impurity (Gini)')
ax2.set_title('(b) Top 20 by Random Forest Importance', fontsize=9, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / "feature_importance.pdf", bbox_inches='tight')
fig.savefig(OUT_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("Generating Figure 4: Confusion Matrices...")
# -------------------------------------------------------------
# Figure 4: Confusion Matrices Side-by-Side
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300)

models_data = [
    ("Baseline (194 Feats)", df_base['TN'].iloc[0], df_base['FP'].iloc[0], df_base['FN'].iloc[0], df_base['TP'].iloc[0]),
    ("MI Selection (97 Feats)", df_mi['TN'].iloc[0], df_mi['FP'].iloc[0], df_mi['FN'].iloc[0], df_mi['TP'].iloc[0]),
    ("RF Importance (97 Feats)", df_rf['TN'].iloc[0], df_rf['FP'].iloc[0], df_rf['FN'].iloc[0], df_rf['TP'].iloc[0])
]

for idx, (title, tn, fp, fn, tp) in enumerate(models_data):
    cm = np.array([[tn, fp], [fn, tp]])
    ax = axes[idx]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
                ax=ax, annot_kws={"size": 8.5})
    
    ax.set_title(title, fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=8)
    if idx == 0:
        ax.set_ylabel('True Label', fontsize=8)
    else:
        ax.set_ylabel('')

plt.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.pdf", bbox_inches='tight')
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("All 4 figures successfully generated in paper/figures/!")
