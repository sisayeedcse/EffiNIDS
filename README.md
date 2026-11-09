# EffiNIDS — Efficient Network Intrusion Detection with UNSW-NB15

> A research project exploring how **feature selection** can make machine-learning-based network intrusion detection faster and leaner — without sacrificing accuracy.

---

## 📖 What Is This Project?

Network Intrusion Detection Systems (NIDS) monitor network traffic and flag malicious activity. Modern NIDS rely on machine learning, but those models can be slow and resource-hungry when given dozens of raw features to process.

This project asks a simple question:

> **Can we safely discard half the features and still detect attacks just as well?**

We test that idea on the well-known **UNSW-NB15 dataset** — a realistic network traffic benchmark created by UNSW Sydney — and compare three approaches:

1. **Baseline** — use every available feature
2. **Mutual Information selection** — keep only the features most statistically related to the attack label
3. **Random Forest Importance selection** — let an initial model vote on which features matter most

Spoiler: cutting 50% of features actually makes the model *slightly better*.

---

## 🧩 The Dataset — UNSW-NB15

| Property | Value |
|---|---|
| Source | UNSW Sydney Cyber Range Lab |
| Task | Binary classification: **Normal** (0) vs **Attack** (1) |
| Raw features | 42 (mix of numeric and categorical) |
| After encoding | 194 (categorical features one-hot encoded) |
| Training records | ~175,000 |
| Test records | 82,332 (37,000 normal · 45,332 attack) |

**Top discriminative features** (by Mutual Information score):

| Rank | Feature | What it represents |
|---|---|---|
| 1 | `sbytes` | Source-to-destination byte count |
| 2 | `sttl` | Source IP Time-To-Live value |
| 3 | `dbytes` | Destination-to-source byte count |
| 4 | `ct_state_ttl` | Connection state + TTL group |
| 5 | `dttl` | Destination TTL value |
| 6 | `rate` | Packet rate |
| 7 | `sload` | Source bits per second |
| 8 | `dur` | Connection duration |

---

## 🔬 Experiments & Results

All experiments use a **Random Forest** classifier (scikit-learn).

### Model Comparison on the Full Test Set

| Model | Features Used | Accuracy | Attack Recall | Attack F1 | False Positive Rate |
|---|---|---|---|---|---|
| Baseline RF | 194 (all) | 87.09% | 98.63% | 89.38% | 27.05% |
| Mutual Information RF | ~97 | 87.15% | 98.55% | 89.41% | 26.82% |
| **RF Feature Importance** | **97 (50% cut)** | **87.19%** | **98.56%** | **89.45%** | **26.73%** |

**Key takeaway:** Removing half the features causes zero accuracy drop — it actually nudges every metric in the right direction.

---

### Robustness: Can the Model Handle Attack Patterns It Has Never Seen?

We split the test set into flows whose feature patterns appeared in training ("seen") and those that did not ("unseen").

| Subset | Records | Accuracy | Attack F1 |
|---|---|---|---|
| **Full test set** | 82,332 | 87.09% | 89.38% |
| Seen attack patterns | 8,541 | 94.43% | 96.73% |
| **Unseen attack patterns** | 73,791 | 86.24% | 88.12% |

89.6% of test traffic involved attack patterns the model had never encountered — and it still correctly identified 88% of them.

---

### Pattern Consistency: What Happens With Ambiguous Data?

Some records share the exact same feature values but carry **different labels** in the dataset — an attack according to one source, normal according to another. These "conflicting" patterns reveal a labelling ambiguity in UNSW-NB15 itself.

| Group | Records | Accuracy | Attack F1 | Notes |
|---|---|---|---|---|
| **Seen-Consistent** | 8,015 | **100%** | **100%** | Same pattern → same label: trivially learnable |
| **Seen-Conflicting** | 526 | 9.5% | 2.9% | Dataset ambiguity — no model can resolve this |
| **Unseen** | 73,791 | 86.2% | 88.1% | True generalisation challenge |

The ~526 conflicting records are a **data quality issue**, not a model failure. All three models perform identically poorly on them.

---

## ⚙️ Setup & Running the Notebook

### Prerequisites

- Python 3.14 (the `.venv` was built with this version)
- The raw dataset Excel files in `dataset/raw/` (not committed to git due to size)

### 1 — Activate the virtual environment

```powershell
# Windows PowerShell
cd d:\UNSW-NB15-Research
.\.venv\Scripts\Activate.ps1
```

### 2 — Launch JupyterLab

```powershell
jupyter lab
```

### 3 — Open the notebook

Navigate to `notebooks/01_dataset_inspection.ipynb` and run cells top-to-bottom.

> **Note:** Loading the Excel files takes a few minutes — they are large. Grab a coffee. ☕

### Installed packages (key ones)

| Package | Version | Purpose |
|---|---|---|
| `pandas` | 3.1.0 | Data loading & manipulation |
| `numpy` | 2.5.4 | Numerical operations |
| `scikit-learn` | 1.9.1 | Random Forest, feature selection, metrics |
| `matplotlib` | 3.11.2 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualisations |
| `openpyxl` | 3.1.5 | Reading `.xlsx` files |
| `joblib` | 1.6.0 | Saving/loading trained models |
| `jupyterlab` | 4.6.5 | Interactive notebook environment |

---

## 📊 Saved Results

All result CSVs are already committed to `results/metrics/` — you do not need to re-run training to browse them.

```powershell
# Quick look at the main results in PowerShell
Get-Content results\metrics\baseline_results.csv
Get-Content results\metrics\robustness_seen_unseen_results.csv
```

---

## 📝 Research Output

The findings from this project were written up as a short paper:

> **EffiNIDS: Efficient Feature Selection for Network Intrusion Detection**
> Submitted to the 2026 Symposium — see `abstract/EffiNIDS_Symposium2026.pdf`

---

## 🚧 Current Limitations & What's Next

| Area | Current State | Potential Improvement |
|---|---|---|
| False Positive Rate | ~27% — high for production use | Tune decision threshold; try cost-sensitive learning |
| Visualisations | `results/figures/` is empty | Add confusion matrices, ROC curves, feature importance plots |
| Code organisation | Everything in one notebook | Refactor into reusable scripts in `src/` |
| Model variety | Only Random Forest tested | Benchmark XGBoost, LightGBM, neural networks |
| Classification depth | Binary only (normal vs. attack) | Explore 10-class attack category (`attack_cat` column) |
| Dataset quality | 526 ambiguous/conflicting records | Investigate and clean labelling inconsistencies |

---

## 🗂️ Glossary

| Term | Meaning |
|---|---|
| **NIDS** | Network Intrusion Detection System — monitors traffic for malicious activity |
| **UNSW-NB15** | A widely-used benchmark dataset for NIDS research |
| **Feature** | A measurable property of a network flow (e.g. packet size, duration) |
| **Feature Selection** | Choosing only the most useful features to reduce model complexity |
| **Random Forest** | An ensemble of decision trees that vote on a classification |
| **Mutual Information** | A statistical measure of how much a feature tells us about the label |
| **One-hot Encoding** | Converting a categorical value (e.g. `tcp`) into binary indicator columns |
| **False Positive Rate** | Fraction of normal traffic incorrectly flagged as an attack |
| **Recall** | Fraction of actual attacks that were correctly detected |
| **F1 Score** | Harmonic mean of Precision and Recall — balances both |
