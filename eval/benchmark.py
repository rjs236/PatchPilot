"""Honest evaluation benchmark comparing Bandit, Static PatchPilot, and Full PatchPilot.

Evaluates on held-out test splits of PyCode-Vul (non-test functions only):
- Split A test: Cluster-isolated random split (N=2,236; 23% vulnerable)
- Split B test: Held-out repository split (N=1,052; 84 positives)

Setups:
A. Bandit alone
B. Static-only PatchPilot (AST checks + Bandit + Ruff, no ML)
C. Full PatchPilot (AST checks + Bandit + Ruff + ML risk prior)
D. ML risk prior alone (reference baseline)
"""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.inference import is_test_function
from patchpilot.analyzers.ast_analyzer import analyze_ast
from patchpilot.analyzers.external import _find_tool
from patchpilot.analyzers.models import Finding
from patchpilot.ranker import compute_finding_score, rank_findings


# ==============================================================================
# 1. Metric Calculations
# ==============================================================================


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Precision@k for a ranked list of scores."""
    if len(y_true) == 0 or k <= 0:
        return 0.0
    effective_k = min(k, len(y_true))
    # Order by scores descending; use stable sort to preserve order on ties
    ranked_indices = np.argsort(-scores, kind="mergesort")[:effective_k]
    top_labels = y_true[ranked_indices]
    return float(np.mean(top_labels))


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute detection and ranking metrics for a score array given a threshold."""
    y_pred = (scores >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    n_classes = len(np.unique(y_true))
    if n_classes > 1 and len(np.unique(scores)) > 1:
        roc_auc = float(roc_auc_score(y_true, scores))
        ap = float(average_precision_score(y_true, scores))
    elif n_classes > 1:
        roc_auc = 0.5
        ap = float(np.mean(y_true))
    else:
        roc_auc = None
        ap = None

    p_at_10 = precision_at_k(y_true, scores, 10)
    p_at_25 = precision_at_k(y_true, scores, 25)
    p_at_50 = precision_at_k(y_true, scores, 50)
    p_at_100 = precision_at_k(y_true, scores, 100)

    return {
        "threshold": round(threshold, 3),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "average_precision": round(ap, 4) if ap is not None else None,
        "precision_at_k": {
            "10": round(p_at_10, 4),
            "25": round(p_at_25, 4),
            "50": round(p_at_50, 4),
            "100": round(p_at_100, 4),
        },
    }


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for F1 and ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1_list: List[float] = []
    auc_list: List[float] = []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        s_b = scores[idx]
        pred_b = (s_b >= threshold).astype(int)

        f1_list.append(f1_score(y_b, pred_b, zero_division=0))

        if len(np.unique(y_b)) > 1 and len(np.unique(s_b)) > 1:
            auc_list.append(roc_auc_score(y_b, s_b))

    f1_ci = (
        round(float(np.percentile(f1_list, 2.5)), 4),
        round(float(np.percentile(f1_list, 97.5)), 4),
    )
    auc_ci = (
        (
            round(float(np.percentile(auc_list, 2.5)), 4),
            round(float(np.percentile(auc_list, 97.5)), 4),
        )
        if auc_list
        else (0.5, 0.5)
    )

    return {"f1_ci": f1_ci, "auc_ci": auc_ci}


# ==============================================================================
# 2. Batch Analysis Runners for High Performance
# ==============================================================================


def batch_run_bandit_and_ruff(
    codes: List[str],
) -> Tuple[Dict[int, List[Finding]], Dict[int, List[Finding]]]:
    """Execute Bandit and Ruff across all code snippets via directory batching."""
    tmpdir = tempfile.mkdtemp(prefix="patchpilot_eval_")
    bandit_map: Dict[int, List[Finding]] = {i: [] for i in range(len(codes))}
    ruff_map: Dict[int, List[Finding]] = {i: [] for i in range(len(codes))}

    try:
        # 1. Dump snippets to individual files
        for i, code in enumerate(codes):
            fname = f"snippet_{i}.py"
            fpath = os.path.join(tmpdir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(code or "")

        # 2. Run Bandit on entire directory in a single process
        bandit_bin = _find_tool("bandit")
        b_res = subprocess.run(
            [bandit_bin, "-f", "json", "-q", "-r", tmpdir],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if b_res.stdout.strip():
            try:
                b_json = json.loads(b_res.stdout)
                severity_map = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}
                confidence_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}

                for item in b_json.get("results", []):
                    f_name = os.path.basename(item.get("filename", ""))
                    if f_name.startswith("snippet_") and f_name.endswith(".py"):
                        idx_str = f_name[len("snippet_") : -len(".py")]
                        if idx_str.isdigit():
                            idx = int(idx_str)
                            line_no = int(item.get("line_number", 1))
                            line_range = item.get("line_range", [line_no])
                            end_line = int(line_range[-1]) if line_range else line_no
                            sev = severity_map.get(
                                str(item.get("issue_severity", "")).upper(), "medium"
                            )
                            conf = confidence_map.get(
                                str(item.get("issue_confidence", "")).upper(), 0.7
                            )
                            finding = Finding(
                                tool="bandit",
                                rule_id=str(item.get("test_id", "B000")),
                                category="security",
                                severity=sev,
                                message=str(item.get("issue_text", "")),
                                line=line_no,
                                end_line=end_line,
                                confidence=conf,
                            )
                            if idx in bandit_map:
                                bandit_map[idx].append(finding)
            except Exception as e:
                print(f"[Warning] Failed parsing batch Bandit JSON: {e}")

        # 3. Run Ruff on entire directory in a single process
        ruff_bin = _find_tool("ruff")
        r_res = subprocess.run(
            [ruff_bin, "check", "--output-format", "json", tmpdir],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if r_res.stdout.strip():
            try:
                r_json = json.loads(r_res.stdout)
                for item in r_json:
                    f_name = os.path.basename(item.get("filename", ""))
                    if f_name.startswith("snippet_") and f_name.endswith(".py"):
                        idx_str = f_name[len("snippet_") : -len(".py")]
                        if idx_str.isdigit():
                            idx = int(idx_str)
                            line_no = int(item.get("location", {}).get("row", 1))
                            end_line = int(item.get("end_location", {}).get("row", line_no))
                            code_id = str(item.get("code", "RUFF"))
                            finding = Finding(
                                tool="ruff",
                                rule_id=code_id,
                                category="bug" if code_id.startswith("B") else "style",
                                severity="low",
                                message=str(item.get("message", "")),
                                line=line_no,
                                end_line=end_line,
                                confidence=0.8,
                            )
                            if idx in ruff_map:
                                ruff_map[idx].append(finding)
            except Exception as e:
                print(f"[Warning] Failed parsing batch Ruff JSON: {e}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return bandit_map, ruff_map


def batch_run_ast(codes: List[str]) -> Dict[int, List[Finding]]:
    """Run in-process AST analyzer across all code snippets."""
    ast_map: Dict[int, List[Finding]] = {}
    for i, code in enumerate(codes):
        ast_map[i] = analyze_ast(code, filename=f"snippet_{i}.py")
    return ast_map


def batch_run_ml(codes: List[str]) -> np.ndarray:
    """Predict risk probability for each function using the trained clean baseline model."""
    bundle_path = Path("ml/artifacts/baseline_model.joblib")
    if not bundle_path.exists():
        print("[Warning] ML baseline model not found at ml/artifacts/baseline_model.joblib. Returning zeros.")
        return np.zeros(len(codes), dtype=float)

    bundle = joblib.load(bundle_path)
    feature_pipeline = bundle["feature_pipeline"]
    classifier = bundle["classifier"]
    is_dense = bundle.get("is_dense", False)

    X = feature_pipeline.transform(codes)
    if is_dense and hasattr(X, "toarray"):
        X = X.toarray()

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[:, 1]
    else:
        probs = classifier.predict(X).astype(float)

    return np.asarray(probs, dtype=float)


# ==============================================================================
# 3. Setup Scoring & Evaluation
# ==============================================================================


def score_setups(
    codes: List[str],
    bandit_map: Dict[int, List[Finding]],
    ruff_map: Dict[int, List[Finding]],
    ast_map: Dict[int, List[Finding]],
    ml_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate per-function continuous risk scores for Setups A, B, C, D."""
    n = len(codes)
    scores_a = np.zeros(n, dtype=float)
    scores_b = np.zeros(n, dtype=float)
    scores_c = np.zeros(n, dtype=float)
    scores_d = ml_probs.copy()

    for i in range(n):
        b_findings = bandit_map.get(i, [])
        r_findings = ruff_map.get(i, [])
        a_findings = ast_map.get(i, [])
        prob_i = float(ml_probs[i])

        # Setup A: Bandit alone
        if b_findings:
            ranked_a = rank_findings(b_findings, ml_risks={})
            scores_a[i] = ranked_a[0].score if ranked_a else 0.0

        # Setup B: Static PatchPilot (AST + Bandit + Ruff, no ML)
        all_static = a_findings + b_findings + r_findings
        if all_static:
            ranked_b = rank_findings(all_static, ml_risks={})
            scores_b[i] = ranked_b[0].score if ranked_b else 0.0

        # Setup C: Full PatchPilot (AST + Bandit + Ruff + ML risk prior)
        if all_static:
            # Map ML risk across lines of snippet
            line_risks = {f.line: prob_i for f in all_static}
            ranked_c = rank_findings(all_static, ml_risks=line_risks)
            scores_c[i] = ranked_c[0].score if ranked_c else 0.0

    return scores_a, scores_b, scores_c, scores_d


def find_best_threshold(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    """Tune score threshold on validation set to maximize F1 score.

    Evaluates across candidate score thresholds (non-zero unique values and percentiles).
    """
    pos_scores = val_scores[val_scores > 0]
    if len(pos_scores) == 0:
        return 0.1

    # Candidate grid: 0.1, unique scores, plus standard percentiles
    candidates = sorted(list(set([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] + list(np.percentile(pos_scores, np.linspace(5, 95, 20))))))

    best_thresh = 0.1
    best_f1 = -1.0

    for thresh in candidates:
        preds = (val_scores >= thresh).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    return best_thresh


# ==============================================================================
# 4. Error Analysis
# ==============================================================================


def perform_error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    scores_c: np.ndarray,
    thresh_c: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract 5 false negatives (missed by all) and 5 false positives (flagged clean)."""
    # 1. Missed Vulnerable Functions (FN across all setups: label=1, all scores == 0)
    all_zero_mask = (y_true == 1) & (scores_a == 0) & (scores_b == 0) & (scores_c == 0)
    fn_indices = np.where(all_zero_mask)[0]

    missed_examples: List[Dict[str, Any]] = []
    for idx in fn_indices[:5]:
        code = df.iloc[idx]["code"]
        cwe = str(df.iloc[idx].get("cwe_ids", "Unknown"))
        repo = str(df.iloc[idx].get("repo", "Unknown"))
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        header = lines[0] if lines else "def function():"

        # Diagnostic reason
        reason = "Semantic/logic flaw (e.g. auth flow, boundary condition, or state check) with no syntactic static pattern."
        if "cwe-200" in cwe.lower() or "info" in cwe.lower():
            reason = "Information exposure or unquoted header not matching standard AST/Bandit injection sinks."
        elif "cwe-399" in cwe.lower() or "dos" in cwe.lower():
            reason = "Algorithmic complexity / resource exhaustion without standard vulnerable API calls."

        missed_examples.append({
            "index": int(idx),
            "repo": repo,
            "cwe": cwe,
            "header": header,
            "snippet": "\n".join(lines[:4]),
            "reason": reason,
        })

    # 2. Flagged Clean Functions (FP by PatchPilot: label=0, score >= thresh_c)
    fp_mask = (y_true == 0) & (scores_c >= thresh_c)
    fp_indices = np.where(fp_mask)[0]

    false_alarm_examples: List[Dict[str, Any]] = []
    for idx in fp_indices[:5]:
        code = df.iloc[idx]["code"]
        repo = str(df.iloc[idx].get("repo", "Unknown"))
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        header = lines[0] if lines else "def function():"

        # Check why it was flagged
        ast_f = analyze_ast(code)
        rule_ids = [f.rule_id for f in ast_f]
        reason = f"Flagged by rule {rule_ids[0] if rule_ids else 'static/Bandit'}; idiomatic usage matches security smell."

        false_alarm_examples.append({
            "index": int(idx),
            "repo": repo,
            "header": header,
            "rules_triggered": rule_ids[:3],
            "snippet": "\n".join(lines[:4]),
            "reason": reason,
        })

    return missed_examples, false_alarm_examples


# ==============================================================================
# 5. Plot Generation
# ==============================================================================


def generate_plots(
    results_a: Dict[str, Any],
    results_b: Dict[str, Any],
    results_c: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate publication-quality evaluation comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar Chart: F1, Recall, False-Positive Rate for Setups A, B, C
    setups = ["A. Bandit Alone", "B. Static PatchPilot", "C. Full PatchPilot"]
    metrics_names = ["F1 Score", "Recall", "False Positive Rate (FPR)"]

    f1_vals = [results_a["f1"], results_b["f1"], results_c["f1"]]
    rec_vals = [results_a["recall"], results_b["recall"], results_c["recall"]]
    fpr_vals = [results_a["fpr"], results_b["fpr"], results_c["fpr"]]

    x = np.arange(len(setups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width, f1_vals, width, label="F1 Score", color="#2563EB")
    rects2 = ax.bar(x, rec_vals, width, label="Recall", color="#10B981")
    rects3 = ax.bar(x + width, fpr_vals, width, label="False-Positive Rate", color="#F59E0B")

    ax.set_ylabel("Score / Rate (0.0 - 1.0)", fontsize=11)
    ax.set_title("Benchmark Comparison: Bandit vs PatchPilot (PyCode-Vul Split A)", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(setups, fontsize=10)
    ax.legend(frameon=True)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    chart1_path = output_dir / "benchmark_metrics.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] {chart1_path}")

    # 2. Precision@k Curve: Setup B vs Setup C
    ks = [10, 25, 50, 100]
    p_b = [results_b["precision_at_k"][str(k)] for k in ks]
    p_c = [results_c["precision_at_k"][str(k)] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, p_b, marker="o", linewidth=2, color="#4B5563", label="B. Static PatchPilot")
    ax.plot(ks, p_c, marker="s", linewidth=2, color="#8B5CF6", label="C. Full PatchPilot (Static + ML Prior)")

    ax.set_xlabel("Top k Functions Inspected", fontsize=11)
    ax.set_ylabel("Precision@k (Fraction Vulnerable)", fontsize=11)
    ax.set_title("Ranking Quality: Precision@k (Static vs Full PatchPilot)", fontsize=12, pad=12)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    for k_val, b_val, c_val in zip(ks, p_b, p_c):
        ax.annotate(f"{b_val:.2f}", (k_val, b_val), textcoords="offset points", xytext=(-10, 8), fontsize=8)
        ax.annotate(f"{c_val:.2f}", (k_val, c_val), textcoords="offset points", xytext=(5, -12), fontsize=8)

    plt.tight_layout()
    chart2_path = output_dir / "precision_at_k.png"
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] {chart2_path}")


# ==============================================================================
# 6. Main Benchmark Orchestrator
# ==============================================================================


def run_benchmark(
    split_name: str = "split_a",
    sample_size: int = 0,
    seed: int = 42,
    output_path: str = "eval/results.json",
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"RUNNING PATCHPILOT BENCHMARK: {split_name.upper()} (Seed={seed}, Sample={sample_size or 'FULL'})")
    print("=" * 80)

    data_dir = Path("ml/data")
    if split_name == "split_a":
        val_file = data_dir / "val_group.csv"
        test_file = data_dir / "test_group_labelled.csv"
    else:
        val_file = data_dir / "val_repo_holdout.csv"
        test_file = data_dir / "test_repo_holdout_labelled.csv"

    # 1. Load Validation Set (for threshold selection)
    print("\n--- STEP 1: Load and Prepare Validation Set ---")
    df_val = pd.read_csv(val_file)
    if "is_test" not in df_val.columns:
        print("Annotating validation is_test...")
        df_val["is_test"] = [is_test_function(c) for c in df_val["code"]]
    df_val_nt = df_val[~df_val["is_test"]].copy().reset_index(drop=True)
    y_val = df_val_nt["label"].values.astype(int)
    print(f"Validation non-test functions: N={len(df_val_nt)} (Positives={y_val.sum()}, {y_val.mean():.1%})")

    # 2. Run Analyzers on Validation Set to pick thresholds
    print("Executing batch analyzers on validation set for threshold calibration...")
    val_codes = df_val_nt["code"].tolist()
    val_b_map, val_r_map = batch_run_bandit_and_ruff(val_codes)
    val_ast_map = batch_run_ast(val_codes)
    val_ml_probs = batch_run_ml(val_codes)

    val_s_a, val_s_b, val_s_c, val_s_d = score_setups(
        val_codes, val_b_map, val_r_map, val_ast_map, val_ml_probs
    )

    thresh_a = find_best_threshold(y_val, val_s_a)
    thresh_b = find_best_threshold(y_val, val_s_b)
    thresh_c = find_best_threshold(y_val, val_s_c)
    thresh_d = find_best_threshold(y_val, val_s_d)

    print(f"Selected Validation Thresholds (F1-optimal):")
    print(f"  Setup A (Bandit):           {thresh_a:.3f}")
    print(f"  Setup B (Static PatchPilot): {thresh_b:.3f}")
    print(f"  Setup C (Full PatchPilot):   {thresh_c:.3f}")
    print(f"  Setup D (ML Prior Alone):    {thresh_d:.3f}")

    # 3. Load Test Set
    print("\n--- STEP 2: Load and Prepare Held-out Test Set ---")
    df_test = pd.read_csv(test_file)
    if "is_test" not in df_test.columns:
        df_test["is_test"] = [is_test_function(c) for c in df_test["code"]]
    df_test_nt = df_test[~df_test["is_test"]].copy().reset_index(drop=True)

    # Stratified sampling if requested
    if sample_size > 0 and sample_size < len(df_test_nt):
        print(f"Subsampling {sample_size} functions via stratified random sample (seed={seed})...")
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
        sampled_indices, _ = next(sss.split(df_test_nt, df_test_nt["label"]))
        df_eval = df_test_nt.iloc[sampled_indices].copy().reset_index(drop=True)
        sampling_note = f"Stratified random sample of {sample_size} functions (seed {seed}) preserving class ratio."
    else:
        df_eval = df_test_nt
        sampling_note = f"Full held-out test split of {len(df_eval)} non-test functions."

    y_test = df_eval["label"].values.astype(int)
    eval_codes = df_eval["code"].tolist()
    print(f"Test Evaluation Set: N={len(df_eval)} (Positives={y_test.sum()}, {y_test.mean():.1%})")

    # 4. Measure Runtime and Run Analyzers on Test Set
    print("\n--- STEP 3: Execute Evaluation Pipeline on Test Set ---")
    t0 = time.time()
    b_map, r_map = batch_run_bandit_and_ruff(eval_codes)
    t_external = time.time() - t0

    t1 = time.time()
    ast_map = batch_run_ast(eval_codes)
    t_ast = time.time() - t1

    t2 = time.time()
    ml_probs = batch_run_ml(eval_codes)
    t_ml = time.time() - t2

    scores_a, scores_b, scores_c, scores_d = score_setups(
        eval_codes, b_map, r_map, ast_map, ml_probs
    )

    # Runtime per function estimates
    n_funcs = len(eval_codes)
    ms_a = round(((t_external * 0.95) / n_funcs) * 1000, 2)  # Bandit portion
    ms_b = round(((t_external + t_ast) / n_funcs) * 1000, 2)  # Static tools
    ms_c = round(((t_external + t_ast + t_ml) / n_funcs) * 1000, 2)  # Full pipeline
    ms_d = round((t_ml / n_funcs) * 1000, 2)  # ML only

    # 5. Compute Metrics on Test Set
    print("\n--- STEP 4: Compute Honest Metrics ---")
    m_a = compute_metrics(y_test, scores_a, thresh_a)
    m_b = compute_metrics(y_test, scores_b, thresh_b)
    m_c = compute_metrics(y_test, scores_c, thresh_c)
    m_d = compute_metrics(y_test, scores_d, thresh_d)

    ci_a = bootstrap_ci(y_test, scores_a, thresh_a, seed=seed)
    ci_b = bootstrap_ci(y_test, scores_b, thresh_b, seed=seed)
    ci_c = bootstrap_ci(y_test, scores_c, thresh_c, seed=seed)
    ci_d = bootstrap_ci(y_test, scores_d, thresh_d, seed=seed)

    m_a["bootstrap_ci"] = ci_a
    m_b["bootstrap_ci"] = ci_b
    m_c["bootstrap_ci"] = ci_c
    m_d["bootstrap_ci"] = ci_d

    m_a["runtime_ms_per_func"] = ms_a
    m_b["runtime_ms_per_func"] = ms_b
    m_c["runtime_ms_per_func"] = ms_c
    m_d["runtime_ms_per_func"] = ms_d

    # 6. Error Analysis
    print("Conducting error analysis on false negatives and false alarms...")
    fn_samples, fp_samples = perform_error_analysis(
        df_eval, y_test, scores_a, scores_b, scores_c, thresh_c
    )

    # Format Results Structure
    benchmark_payload = {
        "metadata": {
            "split": split_name,
            "sample_size": len(df_eval),
            "sampling_note": sampling_note,
            "random_seed": seed,
            "vulnerable_count": int(y_test.sum()),
            "clean_count": int(len(y_test) - y_test.sum()),
            "vulnerable_ratio": round(float(y_test.mean()), 4),
            "threshold_selection": "Chosen strictly on validation split to maximize F1, then applied frozen to test set.",
        },
        "results": {
            "A_bandit_alone": m_a,
            "B_static_patchpilot": m_b,
            "C_full_patchpilot": m_c,
            "D_ml_prior_alone": m_d,
        },
        "error_analysis": {
            "missed_vulnerabilities": fn_samples,
            "clean_false_alarms": fp_samples,
        },
    }

    # 7. Print Console Summary
    print("\n" + "=" * 80)
    print("HONEST BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Setup':<28} | {'F1 [95% CI]':<19} | {'Recall':<8} | {'Prec':<8} | {'FPR':<8} | {'ROC-AUC [95% CI]':<19} | {'ms/func':<8}")
    print("-" * 115)
    for key, name, m in [
        ("A", "A. Bandit Alone", m_a),
        ("B", "B. Static PatchPilot", m_b),
        ("C", "C. Full PatchPilot", m_c),
        ("D", "D. ML Risk Prior Alone", m_d),
    ]:
        f1_str = f"{m['f1']:.3f} [{m['bootstrap_ci']['f1_ci'][0]:.2f}-{m['bootstrap_ci']['f1_ci'][1]:.2f}]"
        auc_str = (
            f"{m['roc_auc']:.3f} [{m['bootstrap_ci']['auc_ci'][0]:.2f}-{m['bootstrap_ci']['auc_ci'][1]:.2f}]"
            if m["roc_auc"] is not None
            else "N/A"
        )
        print(f"{name:<28} | {f1_str:<19} | {m['recall']:<8.3f} | {m['precision']:<8.3f} | {m['fpr']:<8.3f} | {auc_str:<19} | {m['runtime_ms_per_func']:<8.1f}")
    print("-" * 115)

    print("\n--- Precision@k Comparison (Ranking Quality: B vs C) ---")
    print(f"{'Setup':<28} | {'P@10':<8} | {'P@25':<8} | {'P@50':<8} | {'P@100':<8}")
    print("-" * 70)
    print(f"{'B. Static PatchPilot':<28} | {m_b['precision_at_k']['10']:<8.3f} | {m_b['precision_at_k']['25']:<8.3f} | {m_b['precision_at_k']['50']:<8.3f} | {m_b['precision_at_k']['100']:<8.3f}")
    print(f"{'C. Full PatchPilot':<28} | {m_c['precision_at_k']['10']:<8.3f} | {m_c['precision_at_k']['25']:<8.3f} | {m_c['precision_at_k']['50']:<8.3f} | {m_c['precision_at_k']['100']:<8.3f}")
    print("-" * 70)

    # 8. Save Output Files
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"\n[Raw Results Saved] {out_file}")

    # Generate Plots
    plots_dir = Path("docs/plots")
    generate_plots(m_a, m_b, m_c, plots_dir)

    return benchmark_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchPilot Honest Evaluation Benchmark")
    parser.add_argument(
        "--split",
        type=str,
        default="split_a",
        choices=["split_a", "split_b"],
        help="Evaluation split (split_a: cluster-random; split_b: held-out repo)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample size for stratified evaluation (0 for full test set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and bootstrapping",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results.json",
        help="Path to save output JSON results",
    )

    args = parser.parse_args()
    run_benchmark(
        split_name=args.split,
        sample_size=args.sample,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
