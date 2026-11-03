"""Tests for Phase 3 ML Risk Classifier components.

Covers:
- AST feature extraction and single-function analysis
- Model loading, probability predictions, and graceful fallback
- Mapping function-level risk to individual source code lines
- Integration with Ranker (ml_risk factor) and Pipeline (use_ml flag)
All tests run strictly offline.
"""

from unittest.mock import patch
import pytest

from ml.features import ASTFeatureExtractor, extract_ast_features_single
from ml.inference import (
    RiskPredictor,
    get_default_predictor,
    map_function_risks_to_lines,
    predict_risk,
)
from patchpilot.analyzers.models import Finding
from patchpilot.pipeline import review
from patchpilot.ranker import compute_finding_score, rank_findings


def test_ast_feature_extractor_feature_names():
    extractor = ASTFeatureExtractor()
    names = extractor.get_feature_names_out()
    assert len(names) == 23
    assert "complexity" in names
    assert "has_eval_exec" in names
    assert "has_pickle" in names
    assert "has_subprocess" in names


def test_extract_ast_features_single():
    safe_code = """
def add(a: int, b: int) -> int:
    return a + b
"""
    feat_safe = extract_ast_features_single(safe_code)
    assert feat_safe["complexity"] == 1
    assert feat_safe["arg_count"] == 2
    assert feat_safe["has_eval_exec"] == 0
    assert feat_safe["has_pickle"] == 0

    risky_code = """
def execute_user_command(cmd):
    import subprocess
    import pickle
    eval(cmd)
    subprocess.call(cmd, shell=True)
    pickle.loads(cmd)
"""
    feat_risky = extract_ast_features_single(risky_code)
    assert feat_risky["has_eval_exec"] == 1
    assert feat_risky["has_subprocess"] == 1
    assert feat_risky["has_pickle"] == 1
    assert feat_risky["num_calls"] >= 3


def test_ast_feature_extractor_transform():
    extractor = ASTFeatureExtractor()
    codes = [
        "def f(x):\n    return x * 2\n",
        "def g(x):\n    if x > 0:\n        return eval(x)\n    return None\n",
    ]
    matrix = extractor.transform(codes)
    assert matrix.shape == (2, 23)
    eval_idx = list(extractor.get_feature_names_out()).index("has_eval_exec")
    assert matrix[0, eval_idx] == 0.0
    assert matrix[1, eval_idx] == 1.0


def test_predict_risk_live_model():
    predictor = get_default_predictor()
    if not predictor.is_available:
        pytest.skip("Baseline model artifact not found, skipping live inference test")

    code = """
def safe_greeter(name):
    \"\"\"Return a greeting.\"\"\"
    return f"Hello, {name}!"

def unsafe_deserializer(data_str):
    import pickle
    return pickle.loads(data_str)
"""
    risks = predictor.predict_functions(code)
    assert len(risks) == 2

    # Verify both functions produced valid risk probabilities [0.0, 1.0]
    for func_line, score in risks.items():
        assert 0.0 <= score <= 1.0

    # The unsafe function should score higher risk than the safe greeter
    lines = sorted(list(risks.keys()))
    safe_risk = risks[lines[0]]
    unsafe_risk = risks[lines[1]]
    assert unsafe_risk > safe_risk


def test_predict_risk_fallback_when_unavailable():
    dummy_predictor = RiskPredictor(model_path="ml/artifacts/non_existent_model.joblib")
    assert not dummy_predictor.is_available
    res = dummy_predictor.predict_functions("def foo(): pass")
    assert res == {}


def test_predict_risk_syntax_error():
    predictor = get_default_predictor()
    res = predictor.predict_functions("def broken_syntax( pass")
    # For syntax errors, predictor returns pseudo-function risk or empty dict, never throws
    assert isinstance(res, dict)
    for line, prob in res.items():
        assert 0.0 <= prob <= 1.0


def test_map_function_risks_to_lines():
    code = (
        "# Line 1\n"
        "def f():\n"  # Line 2
        "    x = 1\n"  # Line 3
        "    return x\n"  # Line 4
        "\n"  # Line 5
        "def g():\n"  # Line 6
        "    return 2\n"  # Line 7
    )
    func_risks = {2: 0.85, 6: 0.12}
    line_map = map_function_risks_to_lines(code, func_risks)

    assert line_map[2] == 0.85
    assert line_map[3] == 0.85
    assert line_map[4] == 0.85
    assert line_map[6] == 0.12
    assert line_map[7] == 0.12
    assert 1 not in line_map
    assert 5 not in line_map


def test_ranker_incorporates_ml_risk():
    f1 = Finding(
        tool="bandit",
        rule_id="B301",
        category="security",
        severity="high",
        message="Dangerous pickle",
        line=10,
        end_line=10,
        confidence=0.8,
    )

    f2 = Finding(
        tool="bandit",
        rule_id="B301",
        category="security",
        severity="high",
        message="Dangerous pickle",
        line=20,
        end_line=20,
        confidence=0.8,
    )

    # Score with low vs high ml risk
    score_low = compute_finding_score(f2, ml_risk=0.05)
    score_high = compute_finding_score(f1, ml_risk=0.95)
    assert score_high > score_low


def test_pipeline_with_and_without_ml():
    code = """
def load_data(raw):
    import pickle
    return pickle.loads(raw)
"""
    # With ML enabled
    report_ml = review(code, filename="test.py", use_llm=False, use_ml=True)
    assert "high_risk_functions" in report_ml
    high_risk_funcs = report_ml["high_risk_functions"]
    assert len(high_risk_funcs) >= 1
    assert "line" in high_risk_funcs[0]
    assert "risk" in high_risk_funcs[0]
    assert high_risk_funcs[0]["risk"] >= 0.50

    # With ML disabled
    report_no_ml = review(code, filename="test.py", use_llm=False, use_ml=False)
    assert "high_risk_functions" in report_no_ml
    assert report_no_ml["high_risk_functions"] == []
