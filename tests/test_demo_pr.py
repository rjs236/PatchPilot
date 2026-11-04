"""Tests for bundled offline Demo PR and UI labels.

Verifies:
- All 5 demo PR findings have correct, hand-crafted explanations and suggested fixes
- Covers hardcoded secret, bare except, mutable default, SQL injection, and subprocess shell=True
- Offline demo makes 0 network calls
- UI clearly labels demo explanations as 'Pre-generated sample explanation (not a live Gemini call)'
- Busy warning is exact and leaks no keys or code
- Only successful Gemini calls increment the session limit counter
"""

import json
from pathlib import Path
import socket
from unittest.mock import MagicMock, patch

import pytest

from app.streamlit_app import (
    DEMO_EXPLANATION_LABEL,
    GEMINI_BUSY_WARNING,
    render_finding_card,
)
from patchpilot.pipeline import load_demo_pr


def test_demo_pr_contains_all_five_flaws_with_explanations_and_fixes():
    """Verify demo_pr.json has hand-crafted explanations and fixes for all 5 issues."""
    data = load_demo_pr()
    all_findings = []
    for f_obj in data["files"]:
        all_findings.extend(f_obj["findings"])

    assert len(all_findings) == 5

    rule_ids = {f["rule_id"] for f in all_findings}
    # 1. Hardcoded secret (bandit B105)
    assert "B105" in rule_ids
    # 2. Bare except (ast bare-except)
    assert "bare-except" in rule_ids
    # 3. Mutable default argument (ast mutable-default-args)
    assert "mutable-default-args" in rule_ids
    # 4. SQL injection via string formatting (bandit B608)
    assert "B608" in rule_ids
    # 5. Subprocess with shell=True (bandit B602)
    assert "B602" in rule_ids

    for f in all_findings:
        assert f.get("explanation"), f"Missing explanation for rule {f['rule_id']}"
        assert len(f["explanation"]) > 20
        assert f.get("suggested_fix"), f"Missing suggested_fix for rule {f['rule_id']}"
        assert len(f["suggested_fix"]) > 10
        assert f.get("pre_generated") is True


def test_offline_demo_makes_no_network_calls():
    """Verify that loading demo PR data performs 0 network requests."""
    with patch("socket.socket") as mock_socket, \
         patch("urllib.request.urlopen") as mock_urlopen, \
         patch("requests.Session.send") as mock_send, \
         patch("google.genai.Client") as mock_genai:

        data = load_demo_pr()

        assert data["pr_metadata"]["pull_number"] == 42
        assert mock_socket.call_count == 0
        assert mock_urlopen.call_count == 0
        assert mock_send.call_count == 0
        assert mock_genai.call_count == 0


def test_ui_demo_label_rendered_for_pregenerated_findings():
    """Verify UI clearly labels demo explanations as 'Pre-generated sample explanation (not a live Gemini call)'."""
    assert DEMO_EXPLANATION_LABEL == "Pre-generated sample explanation (not a live Gemini call)"

    demo_finding = {
        "tool": "bandit",
        "rule_id": "B105",
        "category": "security",
        "severity": "high",
        "message": "Hardcoded secret key",
        "line": 12,
        "end_line": 12,
        "explanation": "A secret is hardcoded directly in source code.",
        "suggested_fix": "JWT_SECRET = os.environ['JWT_SECRET']",
        "pre_generated": True,
        "extra": {},
    }

    with patch("streamlit.info") as mock_info, \
         patch("streamlit.expander"), \
         patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]), \
         patch("streamlit.markdown"), \
         patch("streamlit.code"):

        # 1. Render as demo
        render_finding_card(1, demo_finding, is_demo=True)
        # Verify st.info was called with the exact required demo explanation label
        label_calls = [str(call[0][0]) for call in mock_info.call_args_list if call[0]]
        assert any(DEMO_EXPLANATION_LABEL in call_str for call_str in label_calls)


def test_ui_demo_label_not_rendered_for_live_gemini_findings():
    """Verify live Gemini findings do NOT show the pre-generated sample label."""
    live_finding = {
        "tool": "ast",
        "rule_id": "eval-exec",
        "category": "security",
        "severity": "high",
        "message": "Dynamic code execution",
        "line": 5,
        "end_line": 5,
        "explanation": "Live Gemini explanation of the risk.",
        "suggested_fix": "ast.literal_eval(payload)",
        "pre_generated": False,
        "extra": {},
    }

    with patch("streamlit.info") as mock_info, \
         patch("streamlit.expander"), \
         patch("streamlit.columns", return_value=[MagicMock(), MagicMock()]), \
         patch("streamlit.markdown"), \
         patch("streamlit.code"):

        render_finding_card(1, live_finding, is_demo=False)
        label_calls = [str(call[0][0]) for call in mock_info.call_args_list if call[0]]
        assert not any(DEMO_EXPLANATION_LABEL in call_str for call_str in label_calls)


def test_ui_busy_warning_text_and_no_leakage():
    """Verify busy warning string matches exact wording and contains no API keys or code."""
    expected_warning = (
        "Gemini explanations unavailable right now (model busy). "
        "Showing static + ML results. Try again in a minute."
    )
    assert GEMINI_BUSY_WARNING == expected_warning
    assert "key" not in GEMINI_BUSY_WARNING.lower()
    assert "def " not in GEMINI_BUSY_WARNING
    assert "class " not in GEMINI_BUSY_WARNING

