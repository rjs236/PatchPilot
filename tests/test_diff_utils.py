"""Unit tests for diff parsing and line-based finding filtering."""

from patchpilot.analyzers.models import Finding
from patchpilot.diff_utils import (
    filter_findings_by_diff,
    filter_ml_risks_by_diff,
    parse_patch_changed_lines,
)


def test_parse_empty_or_none_patch():
    assert parse_patch_changed_lines(None) == set()
    assert parse_patch_changed_lines("") == set()
    assert parse_patch_changed_lines("   ") == set()


def test_parse_single_hunk():
    patch = (
        "@@ -10,5 +10,6 @@ def foo():\n"
        "     x = 1\n"
        "-    y = 2\n"
        "+    y = 3\n"
        "+    z = 4\n"
        "     return x\n"
    )
    # Line 10: context (' x = 1')
    # Line 11: changed ('+ y = 3')
    # Line 12: changed ('+ z = 4')
    # Line 13: context (' return x')
    changed = parse_patch_changed_lines(patch)
    assert changed == {11, 12}


def test_parse_multiple_hunks():
    patch = (
        "@@ -1,3 +1,4 @@\n"
        " a\n"
        "+b\n"
        " c\n"
        "@@ -15,4 +16,4 @@\n"
        " d\n"
        "-e\n"
        "+f\n"
        " g\n"
    )
    # First hunk:
    # 1: ' a' (context)
    # 2: '+b' (changed)
    # 3: ' c' (context)
    # Second hunk starts at 16:
    # 16: ' d' (context)
    # 17: '+f' (changed)
    # 18: ' g' (context)
    changed = parse_patch_changed_lines(patch)
    assert changed == {2, 17}


def test_parse_deletions_only():
    patch = (
        "@@ -10,4 +10,2 @@\n"
        " a\n"
        "-b\n"
        "-c\n"
        " d\n"
    )
    changed = parse_patch_changed_lines(patch)
    assert changed == set()


def test_parse_additions_only():
    patch = (
        "@@ -0,0 +1,3 @@\n"
        "+line1\n"
        "+line2\n"
        "+line3\n"
    )
    changed = parse_patch_changed_lines(patch)
    assert changed == {1, 2, 3}


def test_parse_single_line_hunk_header_omitted_count():
    patch = (
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    changed = parse_patch_changed_lines(patch)
    assert changed == {1}


def test_parse_no_newline_indicator():
    patch = (
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "+line2\n"
        "\\ No newline at end of file\n"
    )
    changed = parse_patch_changed_lines(patch)
    assert changed == {2}


def test_filter_findings_by_diff_single_line():
    f1 = Finding(
        tool="ast",
        rule_id="r1",
        category="security",
        severity="high",
        message="vuln on changed line",
        line=12,
        end_line=12,
        confidence=0.9,
    )
    f2 = Finding(
        tool="ruff",
        rule_id="r2",
        category="style",
        severity="low",
        message="pre-existing smell",
        line=40,
        end_line=40,
        confidence=0.8,
    )

    retained, hidden_count = filter_findings_by_diff([f1, f2], {11, 12, 13})
    assert retained == [f1]
    assert hidden_count == 1


def test_filter_findings_by_diff_range_overlap():
    # Finding spanning lines 5 to 10
    f_span = Finding(
        tool="ast",
        rule_id="complex",
        category="complexity",
        severity="medium",
        message="complex block",
        line=5,
        end_line=10,
        confidence=0.85,
    )
    f_outside = Finding(
        tool="ast",
        rule_id="naming",
        category="code_smell",
        severity="low",
        message="bad name",
        line=20,
        end_line=22,
        confidence=0.7,
    )

    # Changed lines touch line 7 (within 5..10)
    retained, hidden_count = filter_findings_by_diff([f_span, f_outside], {7, 30})
    assert retained == [f_span]
    assert hidden_count == 1


def test_filter_findings_empty_changed_lines():
    f = Finding(
        tool="ast",
        rule_id="r1",
        category="security",
        severity="high",
        message="issue",
        line=1,
        end_line=1,
        confidence=0.95,
    )
    retained, hidden_count = filter_findings_by_diff([f], set())
    assert retained == []
    assert hidden_count == 1


def test_filter_ml_risks_by_diff():
    ml_risks = {1: 0.85, 2: 0.12, 10: 0.90, 15: 0.44}
    changed_lines = {2, 10, 20}
    filtered = filter_ml_risks_by_diff(ml_risks, changed_lines)
    assert filtered == {2: 0.12, 10: 0.90}
