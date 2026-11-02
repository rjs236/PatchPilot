"""Unit tests for LLM enrichment layer and Gemini fallback logic."""

import json
from unittest.mock import MagicMock, patch

from patchpilot.analyzers.models import Finding
from patchpilot.llm import enrich


def test_enrich_empty_findings():
    assert enrich("code = 1", []) == []


def test_enrich_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    f = Finding(
        tool="ast",
        rule_id="bare-except",
        category="bug",
        severity="medium",
        message="Bare except",
        line=4,
        end_line=4,
        confidence=0.9,
    )
    res = enrich("try:\n  pass\nexcept:\n  pass", [f])

    assert len(res) == 1
    assert "GEMINI_API_KEY" in res[0].extra.get("llm_error", "")
    assert res[0].explanation == ""
    assert res[0].suggested_fix == ""


def test_enrich_successful_mocked_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")

    f1 = Finding(
        tool="ast",
        rule_id="eval-exec",
        category="security",
        severity="high",
        message="Use of eval",
        line=2,
        end_line=2,
        confidence=0.95,
    )

    mock_llm_json = json.dumps(
        {
            "enrichments": [
                {
                    "index": 0,
                    "explanation": "eval() allows arbitrary code execution from untrusted strings.",
                    "suggested_fix": "ast.literal_eval(payload)",
                }
            ]
        }
    )

    mock_response = MagicMock()
    mock_response.text = mock_llm_json

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        res = enrich("cmd = '1'\neval(cmd)", [f1])

        assert len(res) == 1
        assert res[0].explanation == "eval() allows arbitrary code execution from untrusted strings."
        assert res[0].suggested_fix == "ast.literal_eval(payload)"
        assert "llm_error" not in res[0].extra


def test_enrich_api_call_failure_never_breaks_review(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")

    f = Finding(
        tool="ast",
        rule_id="eval-exec",
        category="security",
        severity="high",
        message="Use of eval",
        line=2,
        end_line=2,
        confidence=0.95,
    )

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Quota exceeded or connection error")
        mock_client_cls.return_value = mock_client

        res = enrich("eval('x')", [f])

        assert len(res) == 1
        assert "LLM enrichment failed" in res[0].extra.get("llm_error", "")
        # Review continues uninterrupted
        assert res[0].rule_id == "eval-exec"
