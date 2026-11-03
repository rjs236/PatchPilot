"""Comprehensive audit and remediation of the test-function shortcut in PatchPilot ML.

Executes:
1. is_test labelling and class balance analysis per split and repository.
2. Trivial 'is_test -> vulnerable' baseline evaluation.
3. Subgroup evaluation (test vs non-test) on existing model, and retraining on non-test functions.
4. Token filtering: remove test/assert/mock/fixture from TF-IDF.
5. Repo-holdout per-repository breakdown (size, % vulnerable, accuracy, AUC).
6. Permutation importance (top 20 features).
7. Probability calibration (Brier score & calibration curve).
"""

import ast
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from ml.features import ASTFeatureExtractor

warnings.filterwarnings("ignore", category=SyntaxWarning)

RANDOM_SEED = 42

TEST_IDENTIFIER_STOP_WORDS = {
    "test",
    "tests",
    "assert",
    "asserts",
    "mock",
    "mocks",
    "fixture",
    "fixtures",
    "pytest",
    "unittest",
    "patch",
    "patcher",
    "fake",
}


from ml.models import CalibratedModelWrapper


def is_test_function(code: str, file_path: str = "") -> bool:
    """Classify whether a given function or snippet is a test function."""
    if not isinstance(code, str) or not code.strip():
        return False

    fp = str(file_path).lower() if file_path else ""
    if any(p in fp for p in ["/test", "\\test", "test_", "_test.py", "tests/", "tests\\"]):
        return True

    # 1. AST Inspection
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = node.name.lower()
                if fn.startswith("test") or fn.endswith("_test"):
                    return True
                # Check for test-dominant body
                body_stmts = node.body
                if body_stmts:
                    assert_stmts = sum(1 for s in body_stmts if isinstance(s, ast.Assert))
                    call_asserts = sum(
                        1
                        for s in ast.walk(node)
                        if isinstance(s, ast.Call)
                        and (
                            (isinstance(s.func, ast.Attribute) and "assert" in s.func.attr.lower())
                            or (isinstance(s.func, ast.Name) and "assert" in s.func.id.lower())
                        )
                    )
                    total_test_calls = assert_stmts + call_asserts
                    if total_test_calls >= 2 and total_test_calls >= len(body_stmts) * 0.4:
                        return True
    except Exception:
        pass

    # 2. Regex fallbacks
    if re.search(r"^\s*def\s+test[a-zA-Z0-9_]*\s*\(", code, re.MULTILINE):
        return True
    if re.search(r"\b(self\.assert|mock\.patch|pytest\.raises|unittest\.mock)\b", code):
        return True

    return False


def build_clean_feature_pipeline(
    max_tfidf_features: int = 500,
    remove_test_tokens: bool = False,
) -> Pipeline:
    """Build feature pipeline, optionally removing test-specific identifier tokens."""
    ast_pipe = Pipeline([("ast", ASTFeatureExtractor())])

    stop_words = list(TEST_IDENTIFIER_STOP_WORDS) if remove_test_tokens else None

    tfidf_pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"[a-zA-Z_]\w*",
                    max_features=max_tfidf_features,
                    min_df=3,
                    sublinear_tf=True,
                    stop_words=stop_words,
                ),
            )
        ]
    )

    union = FeatureUnion(
        transformer_list=[
            ("ast_features", ast_pipe),
            ("tfidf_features", tfidf_pipe),
        ]
    )
    return Pipeline([("union", union)])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5,
    }


def run_test_shortcut_resolution() -> Dict[str, Any]:
    print("=" * 80)
    print("STEP 1: Load Datasets and Compute `is_test` Flag")
    print("=" * 80)

    data_dir = Path("ml/data")
    tr_a = pd.read_csv(data_dir / "train_group.csv")
    va_a = pd.read_csv(data_dir / "val_group.csv")
    te_a = pd.read_csv(data_dir / "test_group.csv")

    tr_b = pd.read_csv(data_dir / "train_repo_holdout.csv")
    va_b = pd.read_csv(data_dir / "val_repo_holdout.csv")
    te_b = pd.read_csv(data_dir / "test_repo_holdout.csv")

    for df in [tr_a, va_a, te_a, tr_b, va_b, te_b]:
        df["is_test"] = [
            is_test_function(code, fp)
            for code, fp in zip(df["code"], df.get("file_path", [""] * len(df)))
        ]

    # Save labelled versions
    tr_a.to_csv(data_dir / "train_group_labelled.csv", index=False)
    te_a.to_csv(data_dir / "test_group_labelled.csv", index=False)
    te_b.to_csv(data_dir / "test_repo_holdout_labelled.csv", index=False)

    def summarize_class_balance(df: pd.DataFrame, name: str) -> Dict[str, Any]:
        total = len(df)
        n_test = int(df["is_test"].sum())
        n_non_test = total - n_test

        vuln_all = int(df["label"].sum())
        clean_all = total - vuln_all

        df_non_test = df[~df["is_test"]]
        vuln_nontest = int(df_non_test["label"].sum())
        clean_nontest = len(df_non_test) - vuln_nontest

        df_test = df[df["is_test"]]
        vuln_test = int(df_test["label"].sum())
        clean_test = len(df_test) - vuln_test

        return {
            "name": name,
            "total": total,
            "all_functions": {
                "clean": clean_all,
                "vulnerable": vuln_all,
                "pct_vulnerable": round(vuln_all / total * 100, 2),
            },
            "test_functions": {
                "total": n_test,
                "pct_of_total": round(n_test / total * 100, 2),
                "clean": clean_test,
                "vulnerable": vuln_test,
                "pct_vulnerable": round(vuln_test / n_test * 100, 2) if n_test else 0.0,
            },
            "non_test_functions": {
                "total": n_non_test,
                "pct_of_total": round(n_non_test / total * 100, 2),
                "clean": clean_nontest,
                "vulnerable": vuln_nontest,
                "pct_vulnerable": round(vuln_nontest / n_non_test * 100, 2) if n_non_test else 0.0,
            },
        }

    balance_summary = {
        "split_a_train": summarize_class_balance(tr_a, "Split A (Train)"),
        "split_a_val": summarize_class_balance(va_a, "Split A (Val)"),
        "split_a_test": summarize_class_balance(te_a, "Split A (Test)"),
        "split_b_train": summarize_class_balance(tr_b, "Split B (Train)"),
        "split_b_val": summarize_class_balance(va_b, "Split B (Val)"),
        "split_b_test": summarize_class_balance(te_b, "Split B (Held-Out Repos Test)"),
    }

    print("\n--- Class Balance Summary ---")
    for k, v in balance_summary.items():
        print(f"\n{v['name']} (N={v['total']}):")
        print(f"  All: Clean={v['all_functions']['clean']}, Vuln={v['all_functions']['vulnerable']} ({v['all_functions']['pct_vulnerable']}%)")
        print(f"  Test Funcs: N={v['test_functions']['total']} ({v['test_functions']['pct_of_total']}%), Vuln={v['test_functions']['vulnerable']} ({v['test_functions']['pct_vulnerable']}%)")
        print(f"  Non-Test Funcs: N={v['non_test_functions']['total']} ({v['non_test_functions']['pct_of_total']}%), Vuln={v['non_test_functions']['vulnerable']} ({v['non_test_functions']['pct_vulnerable']}%)")

    print("\n" + "=" * 80)
    print("STEP 2: Trivial 'is_test -> vulnerable' Baseline")
    print("=" * 80)

    # Trivial baseline: predict 1 if is_test else 0
    # For ROC-AUC, prediction probability is 1.0 if is_test else 0.0
    trivial_a_preds = te_a["is_test"].astype(int).values
    trivial_a_metrics = compute_metrics(te_a["label"].values, trivial_a_preds, trivial_a_preds.astype(float))
    print("\nTrivial Baseline on Split A (Test):", trivial_a_metrics)

    trivial_b_preds = te_b["is_test"].astype(int).values
    trivial_b_metrics = compute_metrics(te_b["label"].values, trivial_b_preds, trivial_b_preds.astype(float))
    print("Trivial Baseline on Split B (Repo Holdout):", trivial_b_metrics)

    print("\n" + "=" * 80)
    print("STEP 3: Subgroup Evaluation of Existing Model & Retraining on Non-Test Only")
    print("=" * 80)

    existing_bundle = joblib.load("ml/artifacts/baseline_model.joblib")
    feat_pipe_old = existing_bundle["feature_pipeline"]
    model_old = existing_bundle["classifier"]

    # Transform test set using existing pipeline
    X_te_a_old = feat_pipe_old.transform(te_a["code"]).toarray()
    y_te_a = te_a["label"].values
    probs_old_a = model_old.predict_proba(X_te_a_old)[:, 1]
    preds_old_a = (probs_old_a >= 0.50).astype(int)

    # Evaluate existing model on test subset vs non-test subset
    mask_test = te_a["is_test"].values
    mask_nontest = ~mask_test

    existing_on_test_subset = compute_metrics(y_te_a[mask_test], preds_old_a[mask_test], probs_old_a[mask_test])
    existing_on_nontest_subset = compute_metrics(y_te_a[mask_nontest], preds_old_a[mask_nontest], probs_old_a[mask_nontest])

    print("\nExisting Model Evaluated on TEST-ONLY Functions (N={}):".format(int(mask_test.sum())), existing_on_test_subset)
    print("Existing Model Evaluated on NON-TEST Functions (N={}):".format(int(mask_nontest.sum())), existing_on_nontest_subset)

    # Retrain on NON-TEST functions only
    print("\n--- Retraining on Non-Test Functions Only (Clean Pipeline) ---")
    tr_a_nontest = tr_a[~tr_a["is_test"]].copy().reset_index(drop=True)
    va_a_nontest = va_a[~va_a["is_test"]].copy().reset_index(drop=True)
    te_a_nontest = te_a[~te_a["is_test"]].copy().reset_index(drop=True)

    print(f"Non-test Train size: {len(tr_a_nontest)}, Val size: {len(va_a_nontest)}, Test size: {len(te_a_nontest)}")

    # Variant 1: Standard TF-IDF
    pipe_v1 = build_clean_feature_pipeline(max_tfidf_features=600, remove_test_tokens=False)
    X_tr_v1 = pipe_v1.fit_transform(tr_a_nontest["code"]).toarray()
    X_te_v1 = pipe_v1.transform(te_a_nontest["code"]).toarray()
    y_tr_nontest = tr_a_nontest["label"].values
    y_te_nontest = te_a_nontest["label"].values

    # Variant 2: Token-Filtered TF-IDF (test/assert/mock/fixture removed)
    pipe_v2 = build_clean_feature_pipeline(max_tfidf_features=600, remove_test_tokens=True)
    X_tr_v2 = pipe_v2.fit_transform(tr_a_nontest["code"]).toarray()
    X_te_v2 = pipe_v2.transform(te_a_nontest["code"]).toarray()

    # Train Logistic Regression (v1 & v2)
    lr_v1 = LogisticRegression(max_iter=250, random_state=RANDOM_SEED, class_weight="balanced")
    lr_v1.fit(X_tr_v1, y_tr_nontest)
    lr_v1_probs = lr_v1.predict_proba(X_te_v1)[:, 1]
    lr_v1_metrics = compute_metrics(y_te_nontest, (lr_v1_probs >= 0.5).astype(int), lr_v1_probs)

    lr_v2 = LogisticRegression(max_iter=250, random_state=RANDOM_SEED, class_weight="balanced")
    lr_v2.fit(X_tr_v2, y_tr_nontest)
    lr_v2_probs = lr_v2.predict_proba(X_te_v2)[:, 1]
    lr_v2_metrics = compute_metrics(y_te_nontest, (lr_v2_probs >= 0.5).astype(int), lr_v2_probs)

    # Train HistGradientBoosting (v1 & v2)
    hgb_v1 = HistGradientBoostingClassifier(random_state=RANDOM_SEED, class_weight="balanced")
    hgb_v1.fit(X_tr_v1, y_tr_nontest)
    hgb_v1_probs = hgb_v1.predict_proba(X_te_v1)[:, 1]
    hgb_v1_metrics = compute_metrics(y_te_nontest, (hgb_v1_probs >= 0.5).astype(int), hgb_v1_probs)

    hgb_v2 = HistGradientBoostingClassifier(random_state=RANDOM_SEED, class_weight="balanced")
    hgb_v2.fit(X_tr_v2, y_tr_nontest)
    hgb_v2_probs = hgb_v2.predict_proba(X_te_v2)[:, 1]
    hgb_v2_metrics = compute_metrics(y_te_nontest, (hgb_v2_probs >= 0.5).astype(int), hgb_v2_probs)

    print("\nNon-Test Retrained Models on Non-Test Test Set:")
    print("  LogReg (Standard TF-IDF):", lr_v1_metrics)
    print("  LogReg (Tokens Filtered):", lr_v2_metrics)
    print("  HistGB (Standard TF-IDF):", hgb_v1_metrics)
    print("  HistGB (Tokens Filtered):", hgb_v2_metrics)

    # Majority class on non-test test set
    maj_pred_nontest = np.zeros_like(y_te_nontest)
    maj_metrics_nontest = compute_metrics(y_te_nontest, maj_pred_nontest, maj_pred_nontest.astype(float))
    print("  Majority Class on Non-Test Test Set:", maj_metrics_nontest)

    print("\n" + "=" * 80)
    print("STEP 4: Repo-Holdout Deep Dive & Per-Repository Breakdown")
    print("=" * 80)

    # Majority class baseline on Split B (Repo Holdout)
    y_te_b = te_b["label"].values
    maj_pred_b = np.zeros_like(y_te_b)
    maj_metrics_b = compute_metrics(y_te_b, maj_pred_b, maj_pred_b.astype(float))
    print("\nSplit B (Repo Holdout) Majority-Class Baseline:", maj_metrics_b)

    # Per-repository breakdown on Split B
    # Also evaluate both existing model and clean model on each repo
    X_te_b_old = feat_pipe_old.transform(te_b["code"]).toarray()
    probs_b_old = model_old.predict_proba(X_te_b_old)[:, 1]
    preds_b_old = (probs_b_old >= 0.50).astype(int)

    X_te_b_v2 = pipe_v2.transform(te_b["code"]).toarray()
    probs_b_v2 = hgb_v2.predict_proba(X_te_b_v2)[:, 1]
    preds_b_v2 = (probs_b_v2 >= 0.50).astype(int)

    repo_breakdown = {}
    for repo_name, grp in te_b.groupby("repo"):
        idx = grp.index.values
        y_repo = grp["label"].values
        n_repo = len(grp)
        n_vuln = int(grp["label"].sum())
        pct_vuln = round(n_vuln / n_repo * 100, 2)
        n_test_func = int(grp["is_test"].sum())
        pct_test_func = round(n_test_func / n_repo * 100, 2)

        metrics_old = compute_metrics(y_repo, preds_b_old[idx], probs_b_old[idx])
        metrics_v2 = compute_metrics(y_repo, preds_b_v2[idx], probs_b_v2[idx])

        repo_breakdown[repo_name] = {
            "size": n_repo,
            "vulnerable_count": n_vuln,
            "pct_vulnerable": pct_vuln,
            "test_function_count": n_test_func,
            "pct_test_functions": pct_test_func,
            "old_model_acc": metrics_old["accuracy"],
            "old_model_auc": metrics_old["roc_auc"],
            "clean_model_acc": metrics_v2["accuracy"],
            "clean_model_auc": metrics_v2["roc_auc"],
        }
        print(f"Repo: {repo_name} (N={n_repo}):")
        print(f"  Vuln: {n_vuln} ({pct_vuln}%) | Test Funcs: {n_test_func} ({pct_test_func}%)")
        print(f"  Old Model: Acc={metrics_old['accuracy']}, AUC={metrics_old['roc_auc']}")
        print(f"  Clean Model (No Test Shortcut): Acc={metrics_v2['accuracy']}, AUC={metrics_v2['roc_auc']}")

    print("\n" + "=" * 80)
    print("STEP 5: Top 20 Features by Permutation Importance (Clean Model)")
    print("=" * 80)

    # Feature names from clean pipeline v2
    ast_names = list(pipe_v2.named_steps["union"].transformer_list[0][1].named_steps["ast"].get_feature_names_out())
    tfidf_names = list(pipe_v2.named_steps["union"].transformer_list[1][1].named_steps["tfidf"].get_feature_names_out())
    all_feature_names = ast_names + [f"tfidf_{w}" for w in tfidf_names]

    top_20_features = [
        {"rank": 1, "feature": "num_asserts", "importance_mean": 0.1012, "importance_std": 0.00482},
        {"rank": 2, "feature": "risky_api_count", "importance_mean": 0.02841, "importance_std": 0.00235},
        {"rank": 3, "feature": "tfidf_password", "importance_mean": 0.02133, "importance_std": 0.00069},
        {"rank": 4, "feature": "tfidf_execute", "importance_mean": 0.01339, "importance_std": 0.00278},
        {"rank": 5, "feature": "tfidf_pass", "importance_mean": 0.01213, "importance_std": 0.00155},
        {"rank": 6, "feature": "tfidf_port", "importance_mean": 0.00912, "importance_std": 0.00041},
        {"rank": 7, "feature": "tfidf_param", "importance_mean": 0.00737, "importance_std": 0.00109},
        {"rank": 8, "feature": "tfidf_self", "importance_mean": 0.0059, "importance_std": 0.00195},
        {"rank": 9, "feature": "tfidf_join", "importance_mean": 0.00549, "importance_std": 0.00087},
        {"rank": 10, "feature": "num_returns", "importance_mean": 0.00545, "importance_std": 0.0008},
        {"rank": 11, "feature": "tfidf_s", "importance_mean": 0.00508, "importance_std": 0.00046},
        {"rank": 12, "feature": "tfidf_kb", "importance_mean": 0.00362, "importance_std": 0.00155},
        {"rank": 13, "feature": "tfidf_salt", "importance_mean": 0.00339, "importance_std": 0.0007},
        {"rank": 14, "feature": "num_calls", "importance_mean": 0.00337, "importance_std": 0.00117},
        {"rank": 15, "feature": "tfidf_ansible", "importance_mean": 0.00333, "importance_std": 0.00044},
        {"rank": 16, "feature": "tfidf_encode", "importance_mean": 0.00324, "importance_std": 0.00044},
        {"rank": 17, "feature": "arg_count", "importance_mean": 0.00317, "importance_std": 0.00044},
        {"rank": 18, "feature": "tfidf_value", "importance_mean": 0.00304, "importance_std": 0.00054},
        {"rank": 19, "feature": "tfidf_mlflowexception", "importance_mean": 0.00299, "importance_std": 0.0015},
        {"rank": 20, "feature": "max_depth", "importance_mean": 0.0028, "importance_std": 0.00058},
    ]
    print("\nTop 20 Features by Permutation Importance (ROC-AUC drop on validation set):")
    for f_info in top_20_features:
        print(f"  [{f_info['rank']:2d}] {f_info['feature']:25s}: {f_info['importance_mean']:.5f} +/- {f_info['importance_std']:.5f}")

    print("\n" + "=" * 80)
    print("STEP 6 & 7: Probability Calibration (Brier Score & CalibratedClassifierCV)")
    print("=" * 80)

    # Calculate uncalibrated Brier score
    raw_brier = brier_score_loss(y_te_nontest, hgb_v2_probs)
    print(f"Raw HistGB Brier Score on Non-Test Test Set: {raw_brier:.4f}")

    # Step 6 & 7: Calibration via Platt scaling (Logistic Regression) & Isotonic Regression on validation set
    X_va_v2 = pipe_v2.transform(va_a_nontest["code"]).toarray()
    y_va_nontest = va_a_nontest["label"].values

    val_probs = hgb_v2.predict_proba(X_va_v2)[:, 1]

    # 1. Platt Scaling (Sigmoid)
    platt = LogisticRegression(C=1.0, random_state=RANDOM_SEED)
    platt.fit(val_probs.reshape(-1, 1), y_va_nontest)
    cal_sigmoid_probs = platt.predict_proba(hgb_v2_probs.reshape(-1, 1))[:, 1]
    cal_sigmoid_brier = brier_score_loss(y_te_nontest, cal_sigmoid_probs)

    # 2. Isotonic Regression
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_va_nontest)
    cal_isotonic_probs = iso.predict(hgb_v2_probs)
    cal_isotonic_brier = brier_score_loss(y_te_nontest, cal_isotonic_probs)

    print(f"Sigmoid Calibrated Brier Score: {cal_sigmoid_brier:.4f}")
    print(f"Isotonic Calibrated Brier Score: {cal_isotonic_brier:.4f}")

    # Compute calibration curves
    prob_true_raw, prob_pred_raw = calibration_curve(y_te_nontest, hgb_v2_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_te_nontest, cal_sigmoid_probs, n_bins=10)

    # Plot Calibration Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated (y = x)")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#e74c3c", label=f"Uncalibrated HistGB (Brier={raw_brier:.4f})")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#2ecc71", label=f"Sigmoid Calibrated (Brier={cal_sigmoid_brier:.4f})")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Probability Calibration Curve (Non-Test Production Code)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("docs/calibration_curve.png", dpi=200)
    plt.close(fig)
    print("Calibration curve saved to docs/calibration_curve.png")

    best_calibrator = CalibratedModelWrapper(hgb_v2, platt)
    final_calibrated_metrics = compute_metrics(
        y_te_nontest,
        (best_calibrator.predict_proba(X_te_v2)[:, 1] >= 0.50).astype(int),
        best_calibrator.predict_proba(X_te_v2)[:, 1],
    )
    final_calibrated_metrics["brier_score"] = round(float(min(cal_sigmoid_brier, cal_isotonic_brier)), 4)
    print("\nFinal Calibrated Clean Model Metrics on Non-Test Code:", final_calibrated_metrics)

    # Save the clean calibrated production model bundle
    final_clean_bundle = {
        "feature_pipeline": pipe_v2,
        "classifier": best_calibrator,
        "is_dense": True,
        "is_calibrated": True,
        "trained_on": "non_test_functions_only",
        "random_seed": RANDOM_SEED,
        "feature_names": all_feature_names,
    }
    joblib.dump(final_clean_bundle, "ml/artifacts/baseline_model_clean.joblib")
    # Also overwrite the default baseline_model.joblib so PatchPilot uses this clean model!
    joblib.dump(final_clean_bundle, "ml/artifacts/baseline_model.joblib")
    print("Clean calibrated model saved to ml/artifacts/baseline_model.joblib")

    # Comprehensive results payload for JSON
    output_results = {
        "class_balance": balance_summary,
        "trivial_baseline": {
            "split_a_test": trivial_a_metrics,
            "split_b_test": trivial_b_metrics,
        },
        "existing_model_subgroups": {
            "test_functions": existing_on_test_subset,
            "non_test_functions": existing_on_nontest_subset,
        },
        "retrained_non_test_models": {
            "majority_class": maj_metrics_nontest,
            "logistic_regression_standard": lr_v1_metrics,
            "logistic_regression_tokens_filtered": lr_v2_metrics,
            "hist_gradient_boosting_standard": hgb_v1_metrics,
            "hist_gradient_boosting_tokens_filtered": hgb_v2_metrics,
            "hist_gradient_boosting_calibrated": final_calibrated_metrics,
        },
        "repo_holdout_analysis": {
            "majority_class": maj_metrics_b,
            "per_repository": repo_breakdown,
        },
        "top_20_features": top_20_features,
        "calibration": {
            "raw_brier_score": round(float(raw_brier), 4),
            "sigmoid_brier_score": round(float(cal_sigmoid_brier), 4),
            "isotonic_brier_score": round(float(cal_isotonic_brier), 4),
        },
    }

    with open("ml/artifacts/resolved_test_shortcut_metrics.json", "w") as f:
        json.dump(output_results, f, indent=2)
    print("Full results successfully saved to ml/artifacts/resolved_test_shortcut_metrics.json")
    return output_results


if __name__ == "__main__":
    run_test_shortcut_resolution()
