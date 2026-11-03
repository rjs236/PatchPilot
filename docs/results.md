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

---

## 9. Full-Pipeline Benchmark Evaluation (Phase 4)

On the main benchmark, Bandit alone has higher F1 (0.768) than full PatchPilot (0.575) and higher recall; static PatchPilot reaches recall 0.905 but with a 49% false-positive rate caused mostly by maintainability rules; once the ML prior is applied with a validation-chosen threshold, full PatchPilot behaves much like the ML model alone. All evaluations were executed with deterministic parameters (seed 42, no LLM calls) on strictly non-test production functions from PyCode-Vul.

> [!WARNING]
> **Bandit Audit Limitation & Benchmark Scope**:
> Our audit revealed that **86.1% of Bandit's true positives on Split A came from generic risky constructs** (such as standard assertions, bare `except:` blocks, `/tmp` paths, pseudo-random calls, or hashing calls), rather than detection of the specific flaw patched in the commit. Consequently, this benchmark measures statistical agreement with commit-mined labels rather than detection of the specific vulnerabilities themselves.

### 9.1 Evaluation Setups Compared

1. **Setup A (Bandit Alone)**: Standalone Bandit execution with confidence-weighted scoring.
2. **Setup B (Static PatchPilot)**: Custom AST rules + Bandit + Ruff linting with deterministic ranker scoring (no ML risk prior).
3. **Setup C (Full PatchPilot)**: Custom AST rules + Bandit + Ruff linting + ML risk prior modulation ($1 + \text{MLRisk}$).
4. **Setup D (ML Risk Prior Alone)**: Standalone gradient-boosting risk score treated directly as a binary detector (reference baseline).

> [!IMPORTANT]
> **Threshold Selection Protocol**:
> All decision thresholds were selected on the validation partition to maximize F1 (Split A val $N=1,422$; Split B val $N=1,616$) and then applied frozen to the held-out test sets without tuning.
> - Split A test thresholds: Bandit $\ge 0.100$, Static $\ge 0.500$, Full $\ge 2.858$, ML Prior $\ge 0.213$.
> - Split B test thresholds: Bandit $\ge 0.100$, Static $\ge 1.000$, Full $\ge 2.058$, ML Prior $\ge 0.320$.

### 9.2 Primary Benchmark: Split A Non-Test ($N=2,236$; 516 Vulnerable, 23.1%)

| Setup | Precision | Recall | F1 Score [95% CI] | FPR (Clean) | ROC-AUC [95% CI] | PR-AUC | Runtime (ms/func) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Bandit Alone** | **0.8198** | 0.7229 | **0.7683 [0.7365, 0.7964]** | **0.0477** | 0.8486 [0.8272, 0.8687] | 0.7361 | 43.3 ms |
| **B. Static PatchPilot** | 0.3554 | **0.9050** | 0.5104 [0.4839, 0.5372] | 0.4924 | 0.7612 [0.7385, 0.7829] | 0.4858 | 54.9 ms |
| **C. Full PatchPilot** | 0.6305 | 0.5291 | 0.5753 [0.5369, 0.6126] | 0.0930 | 0.8337 [0.8137, 0.8528] | 0.6501 | 60.4 ms |
| **D. ML Risk Prior Alone** | 0.6082 | 0.5446 | 0.5746 [0.5363, 0.6123] | 0.1052 | **0.8183 [0.7958, 0.8398]** | 0.6466 | **5.4 ms** |

#### Confusion Matrices (Split A Non-Test)
- **Setup A (Bandit Alone)**: $\text{TP}=373, \text{FP}=82, \text{TN}=1638, \text{FN}=143$
- **Setup B (Static PatchPilot)**: $\text{TP}=467, \text{FP}=847, \text{TN}=873, \text{FN}=49$
- **Setup C (Full PatchPilot)**: $\text{TP}=273, \text{FP}=160, \text{TN}=1560, \text{FN}=243$
- **Setup D (ML Prior Alone)**: $\text{TP}=281, \text{FP}=181, \text{TN}=1539, \text{FN}=235$

### 9.3 Supporting Benchmark: Split B Non-Test ($N=1,052$; 84 Vulnerable, 8.0%)

| Setup | Precision | Recall | F1 Score [95% CI] | FPR (Clean) | ROC-AUC [95% CI] | PR-AUC | Runtime (ms/func) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Bandit Alone** | **0.8679** | 0.5476 | 0.6715 [0.5714, 0.7574] | **0.0072** | 0.7710 [0.7181, 0.8235] | 0.5406 | 53.5 ms |
| **B. Static PatchPilot** | 0.2112 | **0.9405** | 0.3450 [0.2891, 0.3992] | 0.3048 | 0.8376 [0.7969, 0.8739] | 0.3314 | 58.0 ms |
| **C. Full PatchPilot** | 0.3317 | 0.7976 | 0.4685 [0.3985, 0.5437] | 0.1395 | 0.9276 [0.9010, 0.9538] | 0.7254 | 60.5 ms |
| **D. ML Risk Prior Alone** | 0.6495 | 0.7500 | **0.6961 [0.6222, 0.7723]** | 0.0351 | **0.9640 [0.9422, 0.9833]** | **0.8329** | **2.5 ms** |

#### Confusion Matrices (Split B Non-Test)
- **Setup A (Bandit Alone)**: $\text{TP}=46, \text{FP}=7, \text{TN}=961, \text{FN}=38$
- **Setup B (Static PatchPilot)**: $\text{TP}=79, \text{FP}=295, \text{TN}=673, \text{FN}=5$
- **Setup C (Full PatchPilot)**: $\text{TP}=67, \text{FP}=135, \text{TN}=833, \text{FN}=17$
- **Setup D (ML Prior Alone)**: $\text{TP}=63, \text{FP}=34, \text{TN}=934, \text{FN}=21$

### 9.4 Ranking Quality: Precision@k Comparison (Setup B vs Setup C)

Evaluating whether the top-$k$ alerts presented to a developer contain genuine vulnerabilities:

| Benchmark Split | Setup | P@10 | P@25 | P@50 | P@100 |
|---|---|:---:|:---:|:---:|:---:|
| **Split A Non-Test** ($N=2,236$) | **Setup B (Static PatchPilot)** | 1.000 | 1.000 | 1.000 | **0.960** |
| **Split A Non-Test** ($N=2,236$) | **Setup C (Full PatchPilot)** | 1.000 | 1.000 | 0.980 | 0.900 |
| **Split B Non-Test** ($N=1,052$) | **Setup B (Static PatchPilot)** | 1.000 | 1.000 | 0.900 | 0.450 |
| **Split B Non-Test** ($N=1,052$) | **Setup C (Full PatchPilot)** | 1.000 | 1.000 | 0.800 | **0.590** |

### 9.5 Exploratory: Category-Restricted Analysis (Security + Bug Only)

> [!NOTE]
> **Exploratory, added after reviewing the main results**: The primary results evaluate all findings produced by the analyzers. Because AST complexity rules (`cyclomatic-complexity`, `long-function`, `too-many-args`) and style warnings flag structural characteristics rather than immediate vulnerabilities, this exploratory analysis isolates Setups B and C to only **security** and **bug** category findings. Thresholds were selected on the validation partition only ($B_{\text{sec}}=1.000$, $C_{\text{sec}}=1.271$ on Split A; $B_{\text{sec}}=1.000$, $C_{\text{sec}}=1.140$ on Split B).

| Split | Setup | Precision | Recall | F1 Score [95% CI] | FPR (Clean) | ROC-AUC [95% CI] | PR-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Split A Non-Test** | **B. Static (Sec + Bug)** | **0.7717** | 0.7403 | 0.7557 [0.7240, 0.7856] | **0.0657** | 0.8493 [0.8281, 0.8679] | 0.6690 |
| **Split A Non-Test** | **C. Full (Sec + Bug)** | 0.7495 | **0.7655** | **0.7574 [0.7263, 0.7861]** | 0.0767 | **0.8578 [0.8373, 0.8761]** | **0.7253** |
| **Split B Non-Test** | **B. Static (Sec + Bug)** | **0.6667** | **0.5476** | **0.6013 [0.5063, 0.6879]** | **0.0238** | 0.7524 [0.6984, 0.8061] | 0.4186 |
| **Split B Non-Test** | **C. Full (Sec + Bug)** | 0.6571 | **0.5476** | 0.5974 [0.5033, 0.6829] | 0.0248 | **0.7544 [0.7001, 0.8081]** | **0.4682** |

Restricting findings to security and bug categories on Split A reduces the false-positive rate from 49.2% down to 6.6% for Static (F1 rises from 0.510 to 0.756) and from 9.3% to 7.7% for Full PatchPilot (F1 rises from 0.575 to 0.757).

### 9.6 Bandit Sanity-Check & Limitations

We audited Bandit's detections on Split A ($N=2,236$, 516 vulnerable functions, yielding 373 Bandit TPs and 82 Bandit FPs).

#### Top 10 Bandit Rules Triggered
- **Among True Positives (373 TP functions)**:
  1. `B110` (try_except_pass): 181
  2. `B101` (assert_used): 149
  3. `B108` (hardcoded_tmp_directory): 73
  4. `B311` (random pseudo-random generator): 49
  5. `B602` (subprocess_popen_with_shell_equals_true): 47
  6. `B324` (hashlib insecure md5/sha1): 37
  7. `B105` (hardcoded_password_string): 22
  8. `B608` (hardcoded_sql_expressions): 19
  9. `B104` (hardcoded_bind_all_interfaces): 14
  10. `B307` (eval): 13
- **Among False Positives (82 FP functions, 185 rule triggers)**:
  1. `B608` (hardcoded_sql_expressions): 118
  2. `B603` (subprocess_without_shell_equals_true): 16
  3. `B105` (hardcoded_password_string): 11
  4. `B607` (start_process_with_partial_path): 9
  5. `B108` (hardcoded_tmp_directory): 6
  6. `B113` (request_without_timeout): 5
  7. `B110` (try_except_pass): 5
  8. `B604` (any_other_function_with_shell_equals_true): 2
  9. `B602` (subprocess_popen_with_shell_equals_true): 2
  10. `B321` (ftplib): 1

#### Limitation: Generic Risky-API Presence vs. Actual Vulnerable Change
Of the 373 Bandit True Positives, **321 (86.1%)** were triggered exclusively by generic risky-API presence (`assert`, bare `except:`, `tempfile`/`/tmp`, pseudo-random calls, or md5/sha1 checksums) rather than the actual vulnerability that was fixed in the source commit. Only 52 TPs (13.9%) flagged a line plausibly connected to the actual security flaw (e.g. `subprocess(shell=True)`, SQL concatenation, or `eval`).

#### 10 Sample Bandit True Positives (Inspected Line Context)
1. `getPage(**kwargs)`: Flagged by `B110` on line `except:` (generic exception suppression).
2. `_set_isolation_level(self, isolation_level)`: Flagged by `B101` on line `assert isolation_level in range(1, 5)`.
3. `get_differing_containers(self)`: Flagged by `B104` on line `expected_bound_ports[container_port] = [{'HostIp': "0.0.0.0", 'HostPort': ""}]`.
4. `prepare_arguments_for_docker_build_command(self)`: Flagged by `B311` on line `self._opt_arg("UPGRADE_INVALIDATION_STRING", f"{random.randrange(2**32):x}")`.
5. `munge(self, ds)`: Flagged by `B101` on line `assert isinstance(ds, dict)`.
6. `_gen_pin_drives(pins)`: Flagged by `B103` on line `os.chmod(tdir, 493)`.
7. `get(self, path, include_body=True)`: Flagged by `B101` on line `assert self.request.method == "HEAD"`.
8. `do_REQUEST(self)`: Flagged by `B608` on line `_cursor.execute("SELECT * FROM users WHERE id=%s LIMIT 0, 1" % base64.b64decode("%s===" % self.param))`.
9. `finish(self, chunk=None)`: Flagged by `B101` on line `assert not self._write_buffer, "Cannot send body with 304"`.
10. `saveHistory(completion=None)`: Flagged by `B110` on line `except:`.

> [!WARNING]
> **Bandit Benchmark Limitation**:
> Bandit's high precision and F1 on PyCode-Vul functions partly reflect the presence of generic risky APIs (e.g., standard assertions, bare exception blocks, `/tmp` strings) in complex functions, rather than precise detection of the specific defect patched in the CVE fix commit.

### 9.7 Error Analysis

#### 5 False Negatives (Missed by All Setups)
1. `django/django: alter_db_table(self, model, old_db_table, new_db_table)`: Inspected: Database table rename SQL generation/constraint handling in schema backend; logic flaw with no dangerous AST/Bandit API sink.
2. `django/django: alter_db_table(...)` (variant): Inspected: Database table rename SQL generation/constraint handling in schema backend; logic flaw with no dangerous AST/Bandit API sink.
3. `django/django: supports_transactions(self)`: Inspected: Transaction capability probe executing table rollback test query; feature detection logic with no dangerous AST/Bandit API sink.
4. `django/django: alter_db_table(...)` (variant): Inspected: Database table rename SQL generation/constraint handling in schema backend; logic flaw with no dangerous AST/Bandit API sink.
5. `django/django: get_storage_engine(self, cursor, table_name)`: Inspected: MySQL information_schema table storage engine query; database introspection logic with no dangerous AST/Bandit API sink.

#### 5 False Alarms (Clean Functions Flagged by PatchPilot C)
1. `tornadoweb/tornado: _curl_setup_request(curl, request, buffer, headers)`: Inspected: Flagged for `long-function` (>50 lines) and `cyclomatic-complexity-high` (>15); exhaustive pycurl option configuration in HTTP client backend.
2. `saltstack/salt: latest(name, rev=None, target=None, runas=None, ...)`: Inspected: Flagged for `too-many-args` and `cyclomatic-complexity-high`; Git branch/revision synchronization state handler in Salt module.
3. `ansible/ansible: run(self)`: Inspected: Flagged for `long-function`, `cyclomatic-complexity-high`, and `deep-nesting`; Yum package manager execution loop in Ansible.
4. `saltstack/salt: installed(name, version=None, ...)`: Inspected: Flagged for `syntax-error`; Jinja/template syntax construct inside snippet causing AST parser recovery.
5. `sqlmapproject/sqlmap: checkSqlInjection(place, parameter, value)`: Flagged by `syntax-error`; reason not verified.

### 9.8 Honest Trade-Off Analysis

1. **Bandit vs. PatchPilot**:
   - Bandit alone achieves higher F1 (0.768 vs. 0.575) and a lower false-positive rate (0.048 vs. 0.093) on Split A non-test functions than full PatchPilot.
   - PatchPilot's static analyzer detects more total vulnerabilities than Bandit alone (Recall = 0.905 vs. 0.723, recovering 94 vulnerabilities missed by Bandit), but trades precision (0.355 vs. 0.820) and increases the false positive rate to 0.492 under the all-category setup.
2. **Measured Effect of the ML Risk Prior**:
   - On Split A, adding the ML risk prior changes the overall ROC-AUC ranking score from 0.761 [0.739, 0.783] (Static) to 0.834 [0.814, 0.853] (Full), and raises PR-AUC from 0.486 to 0.650.
   - At the validation-calibrated threshold, Full PatchPilot achieves an FPR of 0.093 compared to 0.492 for Static PatchPilot, while recall drops from 0.905 to 0.529.
3. **Precision@k Ranking**:
   - On Split A, Static PatchPilot achieved P@100 = 0.960 vs. 0.900 for Full PatchPilot.
   - On Split B, Full PatchPilot achieved P@100 = 0.590 vs. 0.450 for Static PatchPilot.

### 9.9 Syntax Errors and Tool Parse Failures Audit

We audited all functions in the held-out test splits for syntax errors and parser failures across AST parsing, Bandit, and Ruff:

| Metric / Failure Type | Split A Non-Test ($N=2,236$) | Split B Non-Test ($N=1,052$) |
|---|:---:|:---:|
| **AST `syntax-error` Findings** | **39 total** (26 Vuln [5.04%], 13 Clean [0.76%]) | **0 total** (0 Vuln [0.0%], 0 Clean [0.0%]) |
| **Bandit Parse Failures** | **5 total** (4 Vuln [0.78%], 1 Clean [0.06%]) | **0 total** (0 Vuln [0.0%], 0 Clean [0.0%]) |
| **Ruff Syntax (`E9`) Failures** | **80 total** (53 Vuln [10.27%], 27 Clean [1.57%]) | **0 total** (0 Vuln [0.0%], 0 Clean [0.0%]) |

#### Root Cause Investigation
- **Is dedenting the cause?** No. Running `textwrap.dedent` on the raw code resolved **0 of the 39 AST syntax errors** on Split A and 0 on Split B.
- **What actually causes the syntax errors?**
  1. **Truncation during dataset mining**: All 39 AST syntax errors in Split A originate from `sqlmapproject/sqlmap` (e.g. `cmdLineParser` and `getPage`), where snippets were cut off mid-identifier, mid-string literal, or mid-expression around line 722 / ~30KB (e.g. `parser.formatter.format_o`, `help=SU`, `unterminated string literal`).
  2. **Template constructs**: In a few snippets from `saltstack/salt`, Jinja template placeholders (`{{ ... }}`) inside Python files caused AST syntax recovery.
- **Pipeline Robustness Added**:
  While dedenting does not fix truncated dataset snippets, user-pasted code (such as indented methods copied from class definitions) frequently raises `IndentationError`. We added automatic `textwrap.dedent` handling to `analyze_ast` and `pipeline.review`, with full test coverage (`test_indented_snippet_is_dedented_and_parsed_successfully` and `test_review_indented_snippet`).

### 9.10 UI Display: Security & Bug Defaults vs. Maintainability Collapsed

To address the 49% false positive rate driven by complexity and style heuristics in full codebases, PatchPilot's Streamlit UI separates findings into two tiers:
1. **Primary View (Default)**: **Security** and **Bug** findings are displayed openly by default with full risk scores, AI root-cause explanations, and suggested remediations.
2. **Maintainability Section (Collapsed Expander)**: **Complexity** and **Style** findings (such as long functions, high cyclomatic complexity, deep nesting, and parameter count warnings) are grouped into a collapsed expander titled:
   `🛠️ Maintainability Issues (N: X Complexity, Y Style)`
   This applies uniformly across both Tab 1 (Snippet Review) and Tab 2 (Pull Request Review).



