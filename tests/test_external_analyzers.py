"""Unit tests for external tool runners (Bandit and Ruff)."""

import json
from unittest.mock import MagicMock, patch

from patchpilot.analyzers.external import run_bandit, run_ruff
from patchpilot.analyzers.models import Finding


def test_run_bandit_empty_code():
    assert run_bandit("") == []
    assert run_bandit("   ") == []


def test_run_ruff_empty_code():
    assert run_ruff("") == []
    assert run_ruff("   ") == []


def test_run_bandit_subprocess_parsing():
    sample_bandit_json = json.dumps(
        {
            "results": [
                {
                    "test_id": "B301",
                    "issue_text": "Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "line_number": 8,
                    "line_range": [8],
                    "test_name": "blacklist",
                }
            ]
        }
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=sample_bandit_json, stderr="", returncode=1)
        findings = run_bandit("import pickle\npickle.loads(data)")

        assert len(findings) == 1
        assert findings[0].tool == "bandit"
        assert findings[0].rule_id == "B301"
        assert findings[0].severity == "high"
        assert findings[0].confidence == 0.9
        assert findings[0].category == "security"
        assert findings[0].line == 8


def test_run_ruff_subprocess_parsing():
    sample_ruff_json = json.dumps(
        [
            {
                "code": "F401",
                "message": "`os` imported but unused",
                "location": {"row": 3, "column": 1},
                "end_location": {"row": 3, "column": 10},
            },
            {
                "code": "B006",
                "message": "Do not use mutable data structures for argument defaults",
                "location": {"row": 7, "column": 5},
                "end_location": {"row": 7, "column": 25},
            },
        ]
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=sample_ruff_json, stderr="", returncode=1)
        findings = run_ruff("import os\ndef f(x=[]): pass")

        assert len(findings) == 2
        f401 = [f for f in findings if f.rule_id == "F401"][0]
        assert f401.tool == "ruff"
        assert f401.category == "bug"
        assert f401.line == 3

        b006 = [f for f in findings if f.rule_id == "B006"][0]
        assert b006.tool == "ruff"
        assert b006.category == "bug"
        assert b006.severity == "medium"
        assert b006.line == 7


def test_external_tools_handle_subprocess_failure_gracefully():
    with patch("subprocess.run", side_effect=OSError("Command not found")):
        assert run_bandit("import os") == []
        assert run_ruff("import os") == []
