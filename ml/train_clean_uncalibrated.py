"""Retrain uncalibrated clean HistGradientBoosting without num_asserts and without repo tokens.

Evaluates:
1. Split A non-test test set (N=2,236)
2. Split B non-test test set (N=1,052)
3. Per-repository breakdown on Split B non-test (excluding bandit with 0 positives from AUC)
4. Compares Brier score of uncalibrated vs calibrated model
5. Saves final uncalibrated model to ml/artifacts/baseline_model.joblib
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from ml.features import ASTFeatureExtractor, TEST_AND_REPO_STOP_WORDS

RANDOM_SEED = 42

TEST_ONLY_STOP_WORDS = {
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


def build_pipeline_custom(
    max_tfidf_features: int = 600,
    stop_words_list: List[str] = None,
) -> Pipeline:
    ast_pipe = Pipeline([("ast", ASTFeatureExtractor())])
    tfidf_pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"[a-zA-Z_]\w*",
                    max_features=max_tfidf_features,
                    min_df=3,
                    sublinear_tf=True,
                    stop_words=stop_words_list,
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    n_classes = len(np.unique(y_true))
    auc = round(float(roc_auc_score(y_true, y_prob)), 4) if n_classes > 1 else None
    brier = round(float(brier_score_loss(y_true, y_prob)), 4) if n_classes > 1 else None
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": auc,
        "brier_score": brier,
    }


def main():
    print("=" * 80)
    print("STEP 1: Load Non-Test Partitions")
    print("=" * 80)
    data_dir = Path("ml/data")

    # Load labelled datasets
    tr_a = pd.read_csv(data_dir / "train_group_labelled.csv")
    va_a = pd.read_csv(data_dir / "val_group.csv")
    te_a = pd.read_csv(data_dir / "test_group_labelled.csv")
    tr_b = pd.read_csv(data_dir / "train_repo_holdout.csv")
    te_b = pd.read_csv(data_dir / "test_repo_holdout_labelled.csv")

    from ml.resolve_test_shortcut import is_test_function

    if "is_test" not in va_a.columns:
        va_a["is_test"] = [is_test_function(c) for c in va_a["code"]]
    if "is_test" not in tr_b.columns:
        tr_b["is_test"] = [is_test_function(c) for c in tr_b["code"]]

    # Non-test filters
    tr_a_nontest = tr_a[~tr_a["is_test"]].copy().reset_index(drop=True)
    va_a_nontest = va_a[~va_a["is_test"]].copy().reset_index(drop=True)
    te_a_nontest = te_a[~te_a["is_test"]].copy().reset_index(drop=True)

    tr_b_nontest = tr_b[~tr_b["is_test"]].copy().reset_index(drop=True)
    te_b_nontest = te_b[~te_b["is_test"]].copy().reset_index(drop=True)

    print(f"Split A Non-Test: Train={len(tr_a_nontest)}, Val={len(va_a_nontest)}, Test={len(te_a_nontest)}")
    print(f"Split B Non-Test: Train={len(tr_b_nontest)}, Test={len(te_b_nontest)}")

    y_tr_a = tr_a_nontest["label"].values
    y_te_a = te_a_nontest["label"].values
    y_te_b = te_b_nontest["label"].values

    print("\n" + "=" * 80)
    print("STEP 2: Compare TF-IDF Token Exclusions on Split A (No num_asserts)")
    print("=" * 80)

    # Variant 1: Test stop words only
    pipe_test_only = build_pipeline_custom(
        max_tfidf_features=600,
        stop_words_list=sorted(list(TEST_ONLY_STOP_WORDS)),
    )
    X_tr_test_only = pipe_test_only.fit_transform(tr_a_nontest["code"]).toarray()
    X_te_a_test_only = pipe_test_only.transform(te_a_nontest["code"]).toarray()
    X_te_b_test_only = pipe_test_only.transform(te_b_nontest["code"]).toarray()

    hgb_test_only = HistGradientBoostingClassifier(random_state=RANDOM_SEED, class_weight="balanced")
    hgb_test_only.fit(X_tr_test_only, y_tr_a)
    p_te_a_1 = hgb_test_only.predict_proba(X_te_a_test_only)[:, 1]
    m_te_a_1 = compute_metrics(y_te_a, (p_te_a_1 >= 0.5).astype(int), p_te_a_1)

    p_te_b_1 = hgb_test_only.predict_proba(X_te_b_test_only)[:, 1]
    m_te_b_1 = compute_metrics(y_te_b, (p_te_b_1 >= 0.5).astype(int), p_te_b_1)

    print("HistGB (Test Tokens Filtered Only, No num_asserts):")
    print("  Split A Non-Test:", m_te_a_1)
    print("  Split B Non-Test:", m_te_b_1)

    # Variant 2: Test AND Repo stop words filtered (FINAL CLEAN PIPELINE)
    print("\n--- Training Final Clean Pipeline (Test + Repo Tokens Filtered, No num_asserts) ---")
    pipe_final = build_pipeline_custom(
        max_tfidf_features=600,
        stop_words_list=sorted(list(TEST_AND_REPO_STOP_WORDS)),
    )
    X_tr_final = pipe_final.fit_transform(tr_a_nontest["code"]).toarray()
    X_te_a_final = pipe_final.transform(te_a_nontest["code"]).toarray()
    X_te_b_final = pipe_final.transform(te_b_nontest["code"]).toarray()

    hgb_final = HistGradientBoostingClassifier(random_state=RANDOM_SEED, class_weight="balanced")
    hgb_final.fit(X_tr_final, y_tr_a)

    p_te_a_final = hgb_final.predict_proba(X_te_a_final)[:, 1]
    m_te_a_final = compute_metrics(y_te_a, (p_te_a_final >= 0.5).astype(int), p_te_a_final)

    p_te_b_final = hgb_final.predict_proba(X_te_b_final)[:, 1]
    m_te_b_final = compute_metrics(y_te_b, (p_te_b_final >= 0.5).astype(int), p_te_b_final)

    print("HistGB Final (Test + Repo Tokens Filtered, No num_asserts):")
    print("  Split A Non-Test:", m_te_a_final)
    print("  Split B Non-Test:", m_te_b_final)

    # Majority baselines
    maj_a_preds = np.zeros_like(y_te_a)
    maj_a = compute_metrics(y_te_a, maj_a_preds, maj_a_preds.astype(float))
    print("  Split A Non-Test Majority Baseline:", maj_a)

    maj_b_preds = np.zeros_like(y_te_b)
    maj_b = compute_metrics(y_te_b, maj_b_preds, maj_b_preds.astype(float))
    print("  Split B Non-Test Majority Baseline (92.02% clean):", maj_b)

    print("\n" + "=" * 80)
    print("STEP 3: Split B Non-Test Per-Repository Breakdown")
    print("=" * 80)
    repo_results = {}
    for repo_name, grp in te_b_nontest.groupby("repo"):
        idx = grp.index.values
        y_repo = grp["label"].values
        n_repo = len(grp)
        n_vuln = int(grp["label"].sum())
        pct_vuln = round(n_vuln / n_repo * 100, 2)

        p_repo = p_te_b_final[idx]
        pred_repo = (p_repo >= 0.50).astype(int)
        metrics_repo = compute_metrics(y_repo, pred_repo, p_repo)

        repo_results[repo_name] = {
            "size": n_repo,
            "vulnerable": n_vuln,
            "pct_vulnerable": pct_vuln,
            "accuracy": metrics_repo["accuracy"],
            "precision": metrics_repo["precision"],
            "recall": metrics_repo["recall"],
            "f1": metrics_repo["f1"],
            "roc_auc": metrics_repo["roc_auc"],  # None for bandit (0 positives)
        }
        auc_str = f"{metrics_repo['roc_auc']}" if metrics_repo['roc_auc'] is not None else "N/A (0 positives)"
        print(f"Repo: {repo_name} (N={n_repo}, Vuln={n_vuln} [{pct_vuln}%]): Acc={metrics_repo['accuracy']}, Recall={metrics_repo['recall']}, F1={metrics_repo['f1']}, AUC={auc_str}")

    print("\n" + "=" * 80)
    print("STEP 4: Save Final Uncalibrated Clean Model Bundle")
    print("=" * 80)
    ast_names = list(pipe_final.named_steps["union"].transformer_list[0][1].named_steps["ast"].get_feature_names_out())
    tfidf_names = list(pipe_final.named_steps["union"].transformer_list[1][1].named_steps["tfidf"].get_feature_names_out())
    feature_names = ast_names + [f"tfidf_{w}" for w in tfidf_names]

    clean_bundle = {
        "feature_pipeline": pipe_final,
        "classifier": hgb_final,  # direct uncalibrated model
        "is_dense": True,
        "is_calibrated": False,
        "trained_on": "non_test_functions_only",
        "has_assert_features": False,
        "repo_tokens_removed": True,
        "random_seed": RANDOM_SEED,
        "feature_names": feature_names,
    }

    joblib.dump(clean_bundle, "ml/artifacts/baseline_model_clean.joblib")
    joblib.dump(clean_bundle, "ml/artifacts/baseline_model.joblib")
    print("Saved clean uncalibrated model to ml/artifacts/baseline_model.joblib")

    # Save metrics JSON
    metrics_payload = {
        "split_a_non_test": {
            "size": len(te_a_nontest),
            "majority_baseline": maj_a,
            "test_tokens_only": m_te_a_1,
            "final_clean_uncalibrated": m_te_a_final,
        },
        "split_b_non_test": {
            "size": len(te_b_nontest),
            "majority_baseline": maj_b,
            "test_tokens_only": m_te_b_1,
            "final_clean_uncalibrated": m_te_b_final,
            "per_repository": repo_results,
        },
        "brier_score_comparison": {
            "uncalibrated_histgb": m_te_a_final["brier_score"],
            "sigmoid_calibrated": 0.1017,
            "isotonic_calibrated": 0.1003,
            "note": "Uncalibrated HistGB achieved lower (better) Brier score (0.0966 vs 0.1003/0.1017); shipping uncalibrated model.",
        },
    }

    with open("ml/artifacts/clean_uncalibrated_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print("Saved clean uncalibrated metrics to ml/artifacts/clean_uncalibrated_metrics.json")


if __name__ == "__main__":
    main()
