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

1. **Recall vs. Precision**:
   - **Bandit** has high precision (0.820) and a very low false-positive rate (4.8%), but misses ~28% of real vulnerabilities (Recall: 0.723).
   - **Static PatchPilot (B)** achieves much higher recall (0.905), catching 90.5% of vulnerabilities. However, broad AST rules and Ruff style checks trigger a 49.2% false-positive rate on clean code, lowering F1 to 0.510.
2. **Impact of the ML Risk Prior (Setup C vs B)**:
   - Full PatchPilot **improves overall ranking discrimination** (ROC-AUC increases from 0.761 [0.74-0.78] to 0.832 [0.81-0.85]).
   - Full PatchPilot slightly improves F1 (0.510 $\to$ 0.519) and precision (0.355 $\to$ 0.364).
   - **Limitation**: In the top 100 findings (Precision@k), Static PatchPilot ranks slightly sharper than Full PatchPilot on Split A (P@100: 0.960 vs 0.900), because the ML prior occasionally promotes clean functions containing generic security smells. On Split B, Full PatchPilot substantially outperforms Static PatchPilot at P@100 (0.610 vs 0.450).

---

## 📁 Artifacts

- `eval/results.json`: Complete machine-readable metrics, confusion matrices, and bootstrap confidence intervals.
- `docs/plots/benchmark_metrics.png`: Bar chart of F1, Recall, and FPR across Setups A, B, and C.
- `docs/plots/precision_at_k.png`: Precision@k curves comparing ranking quality for B vs C.
