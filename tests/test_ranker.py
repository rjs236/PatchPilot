"""Unit tests for finding scoring, de-duplication, and ranking."""

import pytest
from patchpilot.analyzers.models import Finding
from patchpilot.ranker import compute_finding_score, rank_findings


def test_score_calculation_weights():
    # Security (1.5) * High (3.0) * Confidence (1.0) = 4.5
    f_sec_high = Finding(
        tool="bandit",
        rule_id="B101",
        category="security",
        severity="high",
        message="Assert used",
        line=10,
        end_line=10,
        confidence=1.0,
    )
    assert compute_finding_score(f_sec_high) == 4.5

    # Style (0.5) * Low (1.0) * Confidence (0.8) = 0.4
    f_style_low = Finding(
        tool="ast",
        rule_id="comparison-to-none",
        category="style",
        severity="low",
        message="Comparison to None",
        line=12,
        end_line=12,
        confidence=0.8,
    )
    assert compute_finding_score(f_style_low) == 0.4


def test_score_calculation_with_ml_risk():
    f_bug = Finding(
        tool="ast",
        rule_id="mutable-default-arg",
        category="bug",
        severity="medium",
        message="Mutable default",
        line=5,
        end_line=5,
        confidence=1.0,
    )
    # Bug (1.2) * Medium (2.0) * Confidence (1.0) = 2.4
    assert compute_finding_score(f_bug, ml_risk=0.0) == 2.4
    # With ml_risk = 0.5: 2.4 * (1 + 0.5) = 3.6
    assert compute_finding_score(f_bug, ml_risk=0.5) == 3.6


def test_ranker_deduplication_keeps_highest_score():
    # Two findings on line 20 with category 'bug'
    f1 = Finding(
        tool="ruff",
        rule_id="F401",
        category="bug",
        severity="medium",
        message="Unused import",
        line=20,
        end_line=20,
        confidence=0.8,
    )
    f2 = Finding(
        tool="ast",
        rule_id="bare-except",
        category="bug",
        severity="high",
        message="Critical bare except",
        line=20,
        end_line=20,
        confidence=0.9,
    )
    # Different line, category 'security'
    f3 = Finding(
        tool="bandit",
        rule_id="B102",
        category="security",
        severity="high",
        message="Exec used",
        line=35,
        end_line=35,
        confidence=1.0,
    )

    ranked = rank_findings([f1, f2, f3])
    # Line 20 should have only one finding (f2, which has higher score)
    line_20_findings = [f for f in ranked if f.line == 20]
    assert len(line_20_findings) == 1
    assert line_20_findings[0].rule_id == "bare-except"

    # Sorted descending by score
    assert ranked[0].score >= ranked[1].score


def test_ranker_empty_input():
    assert rank_findings([]) == []
