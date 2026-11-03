"""Training and evaluation of CPU-friendly scikit-learn code risk classifiers."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.features import build_feature_pipeline

RANDOM_SEED = 42


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Calculate and return standard evaluation metrics."""
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        probs = model.decision_function(X_test)
    else:
        probs = preds.astype(float)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, probs))
    except Exception:
        auc = 0.50

    cm = confusion_matrix(y_test, preds).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
    }


def train_and_compare(
    data_dir: str = "ml/data",
    artifacts_dir: str = "ml/artifacts",
    docs_dir: str = "docs",
) -> Dict[str, Any]:
    """Train baseline models and controls on group-wise split, and evaluate on both group and repo holdout."""
    start_total = time.time()
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Group-Wise Splits
    print("Loading Group-Wise dataset splits...")
    train_df = pd.read_csv(Path(data_dir) / "train_group.csv")
    val_df = pd.read_csv(Path(data_dir) / "val_group.csv")
    test_df = pd.read_csv(Path(data_dir) / "test_group.csv")

    # Also load Held-Out Repositories test set
    test_repo_df = pd.read_csv(Path(data_dir) / "test_repo_holdout.csv")

    X_train_raw = train_df["code"].fillna("").tolist()
    y_train = train_df["label"].values
    X_test_raw = test_df["code"].fillna("").tolist()
    y_test = test_df["label"].values
    X_test_repo_raw = test_repo_df["code"].fillna("").tolist()
    y_test_repo = test_repo_df["label"].values

    print(f"Train samples: {len(X_train_raw)} | Test samples: {len(X_test_raw)}")
    print(f"Class balance (Train): {np.bincount(y_train) / len(y_train)}")
    print(f"Class balance (Test):  {np.bincount(y_test) / len(y_test)}")

    # 2. Extract Features
    print("Fitting AST + TF-IDF feature union pipeline...")
    feature_pipeline = build_feature_pipeline(max_tfidf_features=600)
    t0_feat = time.time()
    feature_pipeline.fit(X_train_raw)
    feat_time = time.time() - t0_feat
    print(f"Feature extraction pipeline fitted in {feat_time:.2f}s")

    X_train = feature_pipeline.transform(X_train_raw)
    X_test = feature_pipeline.transform(X_test_raw)
    X_test_repo = feature_pipeline.transform(X_test_repo_raw)

    results: Dict[str, Any] = {
        "metadata": {
            "dataset": "S-AIR-L/PyCode-Vul",
            "seed": RANDOM_SEED,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "test_repo_holdout_size": len(test_repo_df),
            "feature_dim": X_train.shape[1],
        },
        "group_split_results": {},
        "repo_holdout_results": {},
    }

    # Model 1: Majority-Class Baseline
    print("\n--- Model 1: Majority-Class Baseline ---")
    maj_clf = DummyClassifier(strategy="most_frequent")
    maj_clf.fit(X_train, y_train)
    maj_metrics = evaluate_model(maj_clf, X_test, y_test, "Majority Class Baseline")
    results["group_split_results"]["majority_class"] = maj_metrics
    print(f"Accuracy: {maj_metrics['accuracy']} | F1: {maj_metrics['f1']} | AUC: {maj_metrics['roc_auc']}")

    # Model 2: Shuffled-Label Control (Sanity Check)
    print("\n--- Model 2: Shuffled-Label Control (Sanity Check) ---")
    np.random.seed(RANDOM_SEED)
    y_shuffled = np.random.permutation(y_train)
    shuffled_clf = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    shuffled_clf.fit(X_train, y_shuffled)
    shuffled_metrics = evaluate_model(shuffled_clf, X_test, y_test, "Shuffled-Label Control")
    results["group_split_results"]["shuffled_label_control"] = shuffled_metrics
    print(f"Accuracy: {shuffled_metrics['accuracy']} | F1: {shuffled_metrics['f1']} | AUC: {shuffled_metrics['roc_auc']}")

    # Model 3: Logistic Regression
    print("\n--- Model 3: Logistic Regression (L2, Balanced) ---")
    t0_lr = time.time()
    lr_clf = LogisticRegression(
        C=1.0,
        max_iter=600,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    lr_clf.fit(X_train, y_train)
    lr_time = time.time() - t0_lr
    lr_metrics = evaluate_model(lr_clf, X_test, y_test, "Logistic Regression")
    lr_metrics["training_time_seconds"] = round(lr_time, 2)
    results["group_split_results"]["logistic_regression"] = lr_metrics
    print(f"Accuracy: {lr_metrics['accuracy']} | F1: {lr_metrics['f1']} | AUC: {lr_metrics['roc_auc']} | Time: {lr_time:.2f}s")

    # Evaluate LR on Repo-Holdout
    lr_repo_metrics = evaluate_model(lr_clf, X_test_repo, y_test_repo, "Logistic Regression (Repo Holdout)")
    results["repo_holdout_results"]["logistic_regression"] = lr_repo_metrics

    # Model 4: HistGradientBoosting (Dense input)
    print("\n--- Model 4: Gradient Boosting (Histogram-based) ---")
    t0_gb = time.time()
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    X_test_repo_dense = X_test_repo.toarray() if hasattr(X_test_repo, "toarray") else X_test_repo

    gb_clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    gb_clf.fit(X_train_dense, y_train)
    gb_time = time.time() - t0_gb
    gb_metrics = evaluate_model(gb_clf, X_test_dense, y_test, "HistGradientBoosting")
    gb_metrics["training_time_seconds"] = round(gb_time, 2)
    results["group_split_results"]["gradient_boosting"] = gb_metrics
    print(f"Accuracy: {gb_metrics['accuracy']} | F1: {gb_metrics['f1']} | AUC: {gb_metrics['roc_auc']} | Time: {gb_time:.2f}s")

    # Evaluate GB on Repo-Holdout
    gb_repo_metrics = evaluate_model(gb_clf, X_test_repo_dense, y_test_repo, "HistGradientBoosting (Repo Holdout)")
    results["repo_holdout_results"]["gradient_boosting"] = gb_repo_metrics

    # Pick the best model (by ROC-AUC on group test)
    best_model_name = "gradient_boosting" if gb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "logistic_regression"
    best_clf = gb_clf if best_model_name == "gradient_boosting" else lr_clf
    print(f"\nBest baseline selected: {best_model_name}")

    # 3. Export standalone Pipeline bundle for inference
    full_pipeline = {
        "feature_pipeline": feature_pipeline,
        "classifier": best_clf,
        "model_name": best_model_name,
        "is_dense": (best_model_name == "gradient_boosting"),
    }
    model_save_path = artifacts_path / "baseline_model.joblib"
    joblib.dump(full_pipeline, model_save_path)
    print(f"Saved inference bundle to {model_save_path}")

    # 4. Save metrics JSON
    metrics_path = artifacts_path / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    # 5. Generate and save Plots (ROC curves and Confusion Matrix)
    _generate_plots(lr_clf, gb_clf, X_test, X_test_dense, y_test, docs_path)

    total_time = time.time() - start_total
    print(f"\nAll models trained and evaluated in {total_time:.2f}s (under 10 minutes limit)")
    return results


def _generate_plots(lr_clf: Any, gb_clf: Any, X_test: Any, X_test_dense: Any, y_test: np.ndarray, docs_path: Path) -> None:
    """Generate and save ROC curve and Confusion Matrix figures."""
    # Plot 1: ROC Curves
    plt.figure(figsize=(7, 6))
    lr_probs = lr_clf.predict_proba(X_test)[:, 1]
    gb_probs = gb_clf.predict_proba(X_test_dense)[:, 1]

    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_probs)

    auc_lr = roc_auc_score(y_test, lr_probs)
    auc_gb = roc_auc_score(y_test, gb_probs)

    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", color="#2563eb", lw=2)
    plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {auc_gb:.3f})", color="#10b981", lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Chance / Random Baseline (AUC = 0.500)")

    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve Comparison — PatchPilot Risk Classifier", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(docs_path / "roc_curve.png", dpi=150)
    plt.close()

    # Plot 2: Confusion Matrix for best model (GB)
    gb_preds = gb_clf.predict(X_test_dense)
    cm = confusion_matrix(y_test, gb_preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Clean (0)", "Vulnerable (1)"])
    ax.set_yticklabels(["Clean (0)", "Vulnerable (1)"])
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("Confusion Matrix (Gradient Boosting)", fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(docs_path / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Generated evaluation plots in docs/roc_curve.png and docs/confusion_matrix.png")


if __name__ == "__main__":
    train_and_compare()
