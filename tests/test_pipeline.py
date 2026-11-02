"""Integration tests for the PatchPilot review pipeline."""

from unittest.mock import patch

from patchpilot.pipeline import review


def test_review_clean_code():
    code = """
def multiply(x: int, y: int) -> int:
    \"\"\"Multiply two integers safely.\"\"\"
    return x * y
"""
    result = review(code, filename="clean_sample.py", use_llm=False)

    assert result["filename"] == "clean_sample.py"
    assert result["lines"] == len(code.splitlines())
    assert result["summary"]["high"] == 0
    assert result["summary"]["medium"] == 0
    assert result["summary"]["low"] == 0
    assert result["summary"]["total"] == 0
    assert result["findings"] == []


def test_review_vulnerable_code():
    code = """
def unsafe_runner(code_str):
    eval(code_str)
"""
    result = review(code, filename="vuln.py", use_llm=False)

    assert result["filename"] == "vuln.py"
    assert result["summary"]["high"] >= 1
    assert result["summary"]["total"] >= 1
    # Check that findings list contains dict representations
    assert isinstance(result["findings"], list)
    rule_ids = [f["rule_id"] for f in result["findings"]]
    assert "eval-exec" in rule_ids or any("B" in r for r in rule_ids)


def test_review_syntax_error_code():
    code = "def invalid_syntax(:\n    pass"
    result = review(code, filename="broken.py", use_llm=False)

    assert result["filename"] == "broken.py"
    assert result["summary"]["high"] >= 1
    assert any(f["rule_id"] == "syntax-error" for f in result["findings"])


def test_review_with_llm_flag():
    code = "eval('1+1')"
    with patch("patchpilot.pipeline.enrich") as mock_enrich:
        mock_enrich.side_effect = lambda c, findings: findings
        result = review(code, use_llm=True)

        assert mock_enrich.called
        assert "summary" in result
        assert "findings" in result
