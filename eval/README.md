# PatchPilot Evaluation Benchmark

This directory contains the reproducible, honest benchmark comparing **PatchPilot** against **Bandit** and its own sub-components on real-world Python functions from the PyCode-Vul dataset.

---

## 🎯 Benchmark Goal

To honestly measure:
1. Does the combined PatchPilot static analysis pipeline detect more real vulnerabilities than a standard Bandit scan?
2. Does the ML risk prior improve finding ranking quality (Precision@k, ROC-AUC, Average Precision)?
3. Where does PatchPilot succeed, and where does it fail or flag false alarms?

---

## 📊 Dataset & Integrity

- **Source**: Held-out test splits from `PyCode-Vul` (mined from CVE security commits across open-source Python repositories).
- **Non-Test Filtering**: All unit test functions (`test_*`, assert-heavy bodies) were pruned beforehand to prevent shortcut learning.
- **Split A (Primary)**: Cluster-isolated random test split ($N = 2,236$; 516 vulnerable, 23.1% positive rate). Twin before/after patches are clustered together so no pair crosses train/val/test.
- **Split B (Supporting)**: Held-out repository test split ($N = 1,052$; 84 vulnerable, 8.0% positive rate). Evaluates cross-codebase performance on unseen repositories.
- **Fairness & Leakage Prevention**: Never touches training or validation functions during test evaluation. All decision thresholds are tuned strictly on the validation set, then applied frozen to the test set.

---

## ⚙️ Setups Compared

All setups evaluate the exact same functions and labels:

| Setup | Description | Method |
|---|---|---|
| **A. Bandit Alone** | Plain Bandit security scanner | Runs Bandit; flags functions with Bandit findings; ranks by Bandit severity $\times$ confidence. |
| **B. Static PatchPilot** | Multi-analyzer static engine without ML | Custom AST rules + Bandit + Ruff; ranks by PatchPilot heuristic severity weights (`ml_risk=0.0`). |
| **C. Full PatchPilot** | Full hybrid review pipeline | AST rules + Bandit + Ruff + ML risk prior in the ranker (`ml_risks={line: prob}`). |
| **D. ML Risk Prior Alone** | Standalone ML classifier | Probability score $P(\text{vulnerable})$ from `HistGradientBoostingClassifier` trained on non-test AST & token features. |

*(Note: Gemini LLM enrichment is excluded from this benchmark because the LLM explains and remediates findings; it does not detect them).*

---

## 🚀 How to Run the Benchmark

### 1. Reproduce Full Split A (Primary Benchmark)
```bash
python eval/benchmark.py --split split_a --seed 42 --output eval/results.json
```

### 2. Reproduce Full Split B (Held-out Repositories)
```bash
python eval/benchmark.py --split split_b --seed 42 --output eval/results_split_b.json
```

### 3. Stratified Random Subsampling
If running on resource-constrained hardware or in CI, use stratified sampling:
```bash
python eval/benchmark.py --split split_a --sample 1000 --seed 42
```

---

## 📈 Key Findings & Honest Trade-offs

1. **Bandit vs. Full PatchPilot**:
   - On the main benchmark Bandit alone achieves higher F1 (0.768 vs. 0.575) and lower false-positive rate (0.048 vs. 0.093) than full PatchPilot, which trades precision for higher recall.
   - PatchPilot's static analyzer catches 90.5% of vulnerabilities (recovering 94 missed by Bandit alone), but has a 49.2% false-positive rate on clean code when all complexity and style categories are included.
2. **Impact of the ML Risk Prior (Setup C vs B)**:
   - Full PatchPilot **improves overall ranking discrimination** (ROC-AUC increases from 0.761 [0.739–0.783] to 0.834 [0.814–0.853], PR-AUC rises from 0.486 to 0.650).
   - On Split B, Full PatchPilot achieves P@100 = 0.590 vs. 0.450 for Static PatchPilot.
3. **Exploratory Category-Restricted Analysis (Security + Bug Only)**:
   - Restricting findings to security and bug categories drops the false-positive rate on Split A from 49.2% to 6.6% for Static (F1 = 0.756) and to 7.7% for Full PatchPilot (F1 = 0.757).
4. **Bandit Sanity Audit**:
   - 86.1% of Bandit True Positives trigger on generic risky API presence (such as `assert`, bare `except:`, `os.chmod`, `random`, or `/tmp`) rather than the actual line fixed in the vulnerability patch commit.

---

## 📁 Artifacts

- `eval/results.json`: Complete machine-readable metrics, confusion matrices, exploratory category-restricted metrics, Bandit sanity audit, and bootstrap confidence intervals for Split A.
- `eval/results_split_b.json`: Complete metrics and confidence intervals for Split B.
- `docs/plots/benchmark_metrics.png`: Bar chart of detection and ranking metrics across setups.
- `docs/plots/precision_at_k.png`: Precision@k curves comparing ranking quality for B vs C.

