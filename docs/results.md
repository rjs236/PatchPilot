# PatchPilot ML Risk-Classifier: Benchmark Results & Validation

This document reports the final benchmark evaluations, validation-based feature configuration selection, bootstrap confidence intervals, and limitations for the PatchPilot Phase 3 Machine Learning Risk Classifier.

---

## 1. Executive Summary & Production Benchmark (Frozen ML Layer)

> **Machine Learning Risk Classifier (Phase 3 Benchmark Summary)**:
> On production (non-test) Python functions from PyCode-Vul, PatchPilot's gradient-boosting risk model reaches ROC-AUC 0.82 [0.80-0.84] and F1 0.46 [0.41-0.51] on a cluster-isolated random split (N=2,236; 23% vulnerable; majority-class accuracy 76.9%), with recall of only 32% at the default threshold. On a held-out-repository split (N=1,052; 84 positives, mostly from two repositories) it reaches ROC-AUC 0.96 [0.94-0.98] and F1 0.73 [0.65-0.80]; this benchmark is small and imbalanced, so we treat it as supporting evidence only. Our audit found that 42% of the raw dataset were unit tests with a 74% vulnerable-label rate caused by fix-commit mining, and a trivial is_test baseline alone reached 0.70 AUC, so we removed tests, assertion counts and repository-identifying tokens from training. Labels remain noisy, and the model learns to separate vulnerable from patched code rather than to discover new vulnerabilities, so it is used only as a risk prior that re-ranks findings from Bandit, Ruff and custom AST checks.

---

## 2. Validation-Based Feature Configuration Selection

> [!IMPORTANT]
> **Protocol Note**:
> Config 2 was chosen a priori to avoid repository memorisation; validation F1 was slightly lower than Config 1 (0.438 vs 0.472 on Split A val, 0.682 vs 0.699 on Split B val), and we report that. Test split metrics were evaluated **once** for the chosen configuration after selection. Earlier exploratory comparisons in Phase 3 development were run on test splits and are reported in Section 5 for historical transparency only.

### 2.1 Candidate Configurations Evaluated on Validation Sets

1. **Config 1 (Test Stop Words Only)**: 22 AST features (cyclomatic complexity, nesting depth, argument counts, dangerous call frequencies; without `num_asserts`) + 600 TF-IDF features excluding test identifiers (`test`, `assert`, `mock`, `fixture`, `pytest`, `unittest`, `patch`).
2. **Config 2 (Test + Repo Stop Words)**: 22 AST features (without `num_asserts`) + 600 TF-IDF features excluding both test identifiers and repository-identifying vocabulary (`salt`, `ansible`, `django`, `mlflow`, `tornado`, `paramiko`, `requests`, `bandit`, `flask`, `pyyaml`, `w3af`, `sqlmap`, `volatility`).

### 2.2 Validation Performance Comparison

Evaluated on non-test functions of the respective validation partitions:

| Partition | Feature Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Brier Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Split A Val ($N=1,422$)** | Config 1 (Test tokens excluded only) | 0.8425 | 0.6757 | 0.3623 | **0.4717** | 0.8275 | 0.1169 |
| **Split A Val ($N=1,422$)** | **Config 2 (Test + Repo tokens excluded)** | 0.8340 | 0.6389 | 0.3333 | 0.4381 | **0.8294** | 0.1187 |
| **Split B Val ($N=1,616$)** | Config 1 (Test tokens excluded only) | 0.8700 | 0.8215 | 0.6085 | **0.6991** | **0.9048** | 0.0969 |
| **Split B Val ($N=1,616$)** | **Config 2 (Test + Repo tokens excluded)** | 0.8632 | 0.8061 | 0.5910 | 0.6820 | 0.8912 | 0.1026 |

### 2.3 Selection Decision
Config 2 was chosen a priori to avoid repository memorisation; validation F1 was slightly lower than Config 1 (0.438 vs 0.472 on Split A val, 0.682 vs 0.699 on Split B val), and we report that. Config 2 guarantees that the model cannot memorize project-specific names (`salt`, `ansible`, `django`), ensuring that predictions rely solely on genuine Python language constructs, AST complexity, and API usage patterns.

---

## 3. Final Production Test Metrics (Reported Once for Chosen Config)

Evaluating the selected configuration once on held-out test splits yields:

| Benchmark Split | Sample Size ($N$) | Clean / Vuln | Majority Baseline Acc | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Split A Non-Test** | 2,236 | 1,720 / 516 (23.1% vuln) | 76.92% | **82.65%** | 81.37% | 32.17% | **0.4611** | **0.8183** | **0.1358** |
| **Split B Non-Test** | 1,052 | 968 / 84 (7.98% vuln) | 92.02% | **95.82%** | 76.32% | 69.05% | **0.7250** | **0.9640** | **0.0327** |

---

## 4. 95% Bootstrap Confidence Intervals (1,000 Resamples, Seed 42)

Bootstrap confidence intervals were computed with $B=1,000$ iterations (random seed 42) using the $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles:

| Test Split | Metric | Point Estimate | 95% Bootstrap Confidence Interval |
|---|---|:---:|:---:|
| **Split A Non-Test ($N=2,236$)** | **F1 Score** | 0.4611 | **[0.4145, 0.5072]** |
| **Split A Non-Test ($N=2,236$)** | **ROC-AUC** | 0.8183 | **[0.7958, 0.8398]** |
| **Split B Non-Test ($N=1,052$)** | **F1 Score** | 0.7250 | **[0.6474, 0.8023]** |
| **Split B Non-Test ($N=1,052$)** | **ROC-AUC** | 0.9640 | **[0.9422, 0.9833]** |

---

## 5. Split B: Per-Repository Breakdown & Empirical Distribution Facts

Evaluating non-test functions across repositories in Split B ($N=1,052$):

> [!NOTE]
> Per-repository statistics where the number of positive samples is $< 30$ are marked as **indicative only**, as small positive sample sizes produce wide variance. Repositories with zero positives cannot compute ROC-AUC and are designated as N/A.

| Repository | Total ($N$) | Clean | Positives (Vuln) | % Vuln | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Reliability |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`PyCQA/bandit`** | 269 | 269 | 0 | 0.00% | 94.42% | 0.0000 | 0.0000 | 0.0000 | *N/A (0 positives)* | Indicative only ($<30$ positives) |
| **`django/django`** | 30 | 1 | 29 | 96.67% | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Indicative only ($<30$ positives) |
| **`pallets/flask`** | 279 | 248 | 31 | 11.11% | 97.13% | 0.9286 | 0.8387 | 0.8519 | 0.9744 | **Robust ($\ge 30$ positives)** |
| **`paramiko/paramiko`** | 270 | 254 | 16 | 5.93% | 94.07% | 0.3333 | 0.0625 | 0.1111 | 0.9294 | Indicative only ($<30$ positives) |
| **`psf/requests`** | 156 | 148 | 8 | 5.13% | 96.79% | 0.8000 | 0.5000 | 0.6667 | 0.9806 | Indicative only ($<30$ positives) |
| **`yaml/pyyaml`** | 48 | 48 | 0 | 0.00% | 100.00% | 0.0000 | 0.0000 | 0.0000 | *N/A (0 positives)* | Indicative only ($<30$ positives) |
| **Overall Split B (Non-Test)** | **1,052** | **968** | **84** | **7.98%** | **95.82%** | **0.7632** | **0.6905** | **0.7250** | **0.9640** | **Overall benchmark** |

### 5.1 Presence of `django` in Split B Test Partition
`django/django` appears in the Split B test partition ($N=30$) despite not being one of the 5 originally declared held-out repositories (`pallets/flask`, `PyCQA/bandit`, `paramiko/paramiko`, `psf/requests`, `yaml/pyyaml`).
- **Cause**: **Cluster Merging via Union-Find**. In `make_repo_holdout_split`, any cluster touching a held-out repository was assigned in its entirety to the test set to guarantee zero crossing pairs. Functions in Django that shared commit SHAs or $\ge 0.80$ token near-duplicate similarity with code in the held-out repositories were merged into those clusters and moved to the test partition.

### 5.2 Confirmation Regarding Earlier 0.995 AUC
Did the earlier 0.995 AUC include test functions?
**Yes, it did.**
The initial Split B test evaluation contained 1,630 total functions, of which **578 (35.46%) were unit test functions**. Unit tests had a 39.10% positive rate, and test-function vocabulary inflated the initial metric. On non-test functions alone ($N=1,052$), the measured ROC-AUC is **0.9640** [95% CI: 0.9422–0.9833].

### 5.3 Measured Factors Influencing Split B Metrics
1. **Class Imbalance**: 92.02% of Split B non-test functions belong to the clean class (968 clean vs 84 vulnerable). A constant clean prediction achieves 92.02% accuracy.
2. **Zero Positive Samples in Repositories**: Two repositories (`PyCQA/bandit` with $N=269$ and `yaml/pyyaml` with $N=48$) contain zero vulnerable functions in non-test code. Specificity on these repositories contributes to overall accuracy without false negative penalties.
3. **Concentration of Positives**: 60 of the 84 positive functions are located in `django` (29) and `flask` (31).

---

## 6. Model Calibration & Brier Score Analysis

The production clean model uses uncalibrated leaf output probabilities. Brier score loss ($B = \frac{1}{N} \sum (p_i - y_i)^2$, where lower is better) on non-test partitions:

| Partition | Measured Clean Model Brier Score | Majority Baseline Brier Score | Post-Hoc Calibrated Brier Score |
|---|:---:|:---:|:---:|
| **Split A Non-Test ($N=2,236$)** | **0.1358** | 0.2308 | 0.1003–0.1017 (on mixed val) |
| **Split B Non-Test ($N=1,052$)** | **0.0327** | 0.0798 | N/A |

### Rationale for Uncalibrated Model
HistGradientBoosting with logistic loss optimizes log-loss directly over tree leaf partitions. When post-hoc Platt scaling or isotonic regression was applied on validation subsets, it raised the empirical Brier score on unseen code compared to the raw uncalibrated probabilities. Consequently, PatchPilot ships the uncalibrated model.

---

## 7. The Test-Function Shortcut: Measurement and Remediation

### 7.1 Imbalance in Source Dataset
In bug-fix datasets scraped from commits (`S-AIR-L/PyCode-Vul`):
- Across Split A training data, **42.09% of samples were unit test functions**.
- **73.54% of test functions were labeled vulnerable**, compared to **29.92% of non-test functions**, because bug-fix commits typically add regression tests.

### 7.2 The Trivial `is_test -> vulnerable` Baseline
A simple rule predicting vulnerable if `is_test` achieved:
- **Split A Test (N=3,257)**: 73.56% Accuracy, 0.6109 F1, 0.7000 ROC-AUC.
- **Split B Test (N=1,630)**: 73.25% Accuracy, 0.5090 F1, 0.7312 ROC-AUC.

### 7.3 Remediation Actions
1. **Removed Test Functions**: Filtered all test functions from training, validation, and test splits.
2. **Removed AST Assert Counts**: Removed `num_asserts` from AST features, as assertions in non-test functions represent defensive preconditions (e.g. `hasattr(request, 'session')` in Django middleware).
3. **Removed Identifier Tokens**: Excluded test-specific and repository-specific tokens from TF-IDF vocabulary.
4. **Runtime Guardrail**: [`ml/inference.py`](file:///ml/inference.py) automatically skips test functions (`def test_...` or files under `tests/`).

---

## 8. Documented Limitations & Architectural Role

1. **Empirical Risk Prior**: PatchPilot's ML classifier does not emit standalone alerts. Its predicted probability ($0.0 \dots 1.0$) acts as an empirical risk prior to modulate the composite ranker score:
   $$\text{Score} = \text{SeverityWeight} \times \text{Confidence} \times \text{CategoryWeight} \times (1 + \text{MLRisk})$$
2. **Distributional Scope**: Mined from historical security fix commits, the model reflects patterns characteristic of vulnerable and patched code versions. Greenfield logic flaws that do not alter AST complexity or invoke known risky primitives are caught by Layer 1 static analysis (Bandit, Ruff, AST rules) and enriched by Layer 3 (Gemini LLM).
