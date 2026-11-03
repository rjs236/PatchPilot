"""Unit tests for evaluation benchmark metric helpers and flagging logic."""

import numpy as np
import pytest

from eval.benchmark import (
    bootstrap_ci,
    compute_metrics,
    find_best_threshold,
    precision_at_k,
)


def test_precision_at_k_perfect_ranking():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    assert precision_at_k(y_true, scores, k=1) == 1.0
    assert precision_at_k(y_true, scores, k=2) == 1.0
    assert precision_at_k(y_true, scores, k=3) == 1.0
    assert precision_at_k(y_true, scores, k=6) == 0.5


def test_precision_at_k_inverted_ranking():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert precision_at_k(y_true, scores, k=1) == 0.0
    assert precision_at_k(y_true, scores, k=2) == 0.0
    assert precision_at_k(y_true, scores, k=4) == 0.5


def test_precision_at_k_bounds_and_empty():
    assert precision_at_k(np.array([]), np.array([]), k=5) == 0.0
    assert precision_at_k(np.array([1, 0]), np.array([0.8, 0.2]), k=0) == 0.0
    assert precision_at_k(np.array([1, 0]), np.array([0.8, 0.2]), k=-1) == 0.0
    # k larger than array
    assert precision_at_k(np.array([1, 0]), np.array([0.8, 0.2]), k=10) == 0.5


def test_compute_metrics_flagging_logic():
    # 4 functions: 2 vulnerable, 2 clean
    y_true = np.array([1, 1, 0, 0])
    # scores: item 0: 3.5, item 1: 1.0, item 2: 2.0, item 3: 0.0
    scores = np.array([3.5, 1.0, 2.0, 0.0])

    # Case 1: Threshold = 1.5 -> flags items 0 and 2
    # Item 0 (y=1, s=3.5) -> TP
    # Item 1 (y=1, s=1.0) -> FN
    # Item 2 (y=0, s=2.0) -> FP
    # Item 3 (y=0, s=0.0) -> TN
    m = compute_metrics(y_true, scores, threshold=1.5)

    assert m["confusion_matrix"]["tp"] == 1
    assert m["confusion_matrix"]["fn"] == 1
    assert m["confusion_matrix"]["fp"] == 1
    assert m["confusion_matrix"]["tn"] == 1

    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["fpr"] == 0.5  # FP / (FP + TN) = 1 / (1 + 1) = 0.5
    assert m["threshold"] == 1.5
    assert "precision_at_k" in m
    assert m["roc_auc"] is not None


def test_bootstrap_ci_shape_and_bounds():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1])

    ci = bootstrap_ci(y_true, scores, threshold=0.5, n_resamples=100, seed=42)

    assert "f1_ci" in ci
    assert "auc_ci" in ci

    f1_low, f1_high = ci["f1_ci"]
    auc_low, auc_high = ci["auc_ci"]

    assert 0.0 <= f1_low <= f1_high <= 1.0
    assert 0.0 <= auc_low <= auc_high <= 1.0

    # Deterministic check with same seed
    ci2 = bootstrap_ci(y_true, scores, threshold=0.5, n_resamples=100, seed=42)
    assert ci == ci2


def test_find_best_threshold():
    y_val = np.array([1, 1, 0, 0, 0])
    # Clear separation at 2.0
    val_scores = np.array([3.0, 2.5, 1.0, 0.5, 0.0])

    best_t = find_best_threshold(y_val, val_scores)
    # Threshold between 1.0 and 2.5 achieves perfect F1=1.0
    preds = (val_scores >= best_t).astype(int)
    assert np.array_equal(preds, y_val)
