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


def test_enrich_retry_on_503_eventual_success(monkeypatch):
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

    mock_llm_json = json.dumps(
        {
            "enrichments": [
                {
                    "index": 0,
                    "explanation": "Dangerous eval usage.",
                    "suggested_fix": "ast.literal_eval(payload)",
                }
            ]
        }
    )
    mock_response = MagicMock()
    mock_response.text = mock_llm_json

    mock_sleep = MagicMock()
    mock_on_retry = MagicMock()

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        # Fail twice with 503, succeed on 3rd attempt
        mock_client.models.generate_content.side_effect = [
            RuntimeError("503 Service Unavailable"),
            RuntimeError("503 model is overloaded"),
            mock_response,
        ]
        mock_client_cls.return_value = mock_client

        res = enrich("eval('x')", [f], on_retry=mock_on_retry, sleep_fn=mock_sleep)

        assert len(res) == 1
        assert res[0].explanation == "Dangerous eval usage."
        assert res[0].suggested_fix == "ast.literal_eval(payload)"
        assert "llm_error" not in res[0].extra
        assert mock_client.models.generate_content.call_count == 3
        assert mock_sleep.call_count == 2
        assert mock_on_retry.call_count == 2

        # Check delays match backoff: first ~2s, second ~4s
        delay1 = mock_sleep.call_args_list[0][0][0]
        delay2 = mock_sleep.call_args_list[1][0][0]
        assert 2.0 <= delay1 <= 3.5
        assert 4.0 <= delay2 <= 5.5


def test_enrich_retry_on_503_exhausts_max_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")

    f = Finding(
        tool="bandit",
        rule_id="B105",
        category="security",
        severity="high",
        message="Hardcoded secret",
        line=5,
        end_line=5,
        confidence=0.95,
    )

    mock_sleep = MagicMock()
    mock_on_retry = MagicMock()

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")
        mock_client_cls.return_value = mock_client

        res = enrich("secret = '123'", [f], on_retry=mock_on_retry, sleep_fn=mock_sleep)

        # 1 initial + 5 retries = 6 total calls
        assert mock_client.models.generate_content.call_count == 6
        assert mock_sleep.call_count == 5
        assert mock_on_retry.call_count == 5

        # Check all 5 exponential backoff base delays (2, 4, 8, 12, 16)
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        expected_bases = [2.0, 4.0, 8.0, 12.0, 16.0]
        for delay, base in zip(delays, expected_bases):
            assert base <= delay <= base + 1.5

        # Total sleep capped at <= 45s
        assert sum(delays) <= 45.0

        # Findings unchanged and error recorded in extra["llm_error"]
        assert len(res) == 1
        assert res[0].explanation == ""
        assert res[0].suggested_fix == ""
        assert "503 Service Unavailable" in res[0].extra.get("llm_error", "")


def test_enrich_retry_on_429_and_500(monkeypatch):
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

    mock_llm_json = json.dumps(
        {
            "enrichments": [
                {
                    "index": 0,
                    "explanation": "Resolved after backoff.",
                    "suggested_fix": "safe_call()",
                }
            ]
        }
    )
    mock_response = MagicMock()
    mock_response.text = mock_llm_json
    mock_sleep = MagicMock()

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        # Fail with 429, then 500, then succeed
        mock_client.models.generate_content.side_effect = [
            RuntimeError("429 Too Many Requests"),
            RuntimeError("500 Internal Server Error"),
            mock_response,
        ]
        mock_client_cls.return_value = mock_client

        res = enrich("eval('x')", [f], sleep_fn=mock_sleep)

        assert mock_client.models.generate_content.call_count == 3
        assert mock_sleep.call_count == 2
        assert res[0].explanation == "Resolved after backoff."
        assert "llm_error" not in res[0].extra


def test_enrich_no_retry_on_400_bad_request(monkeypatch):
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

    mock_sleep = MagicMock()

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("400 Bad Request: Invalid argument")
        mock_client_cls.return_value = mock_client

        res = enrich("eval('x')", [f], sleep_fn=mock_sleep)

        # Fails immediately without retry
        assert mock_client.models.generate_content.call_count == 1
        assert mock_sleep.call_count == 0
        assert "400 Bad Request" in res[0].extra.get("llm_error", "")


def test_enrich_default_model_gemini_35_flash(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

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

    mock_response = MagicMock()
    mock_response.text = '{"enrichments": []}'

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        enrich("eval('x')", [f])

        # Verify model used is gemini-3.5-flash
        called_model = mock_client.models.generate_content.call_args[1]["model"]
        assert called_model == "gemini-3.5-flash"

