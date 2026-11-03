"""External static analysis runners for Bandit and Ruff with JSON output."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List

from patchpilot.analyzers.models import Finding


def _find_tool(name: str) -> str:
    """Resolve executable path, preferring the current Python environment's bin/Scripts."""
    py_dir = Path(sys.executable).parent
    for suffix in [".exe", ""]:
        cand = py_dir / f"{name}{suffix}"
        if cand.is_file():
            return str(cand)
    found = shutil.which(name)
    return found or name


def run_bandit(code: str, timeout: float = 10.0) -> List[Finding]:
    """Execute Bandit security scanner on code snippet via temporary file.

    Returns empty list on tool failure or timeout.
    """
    if not code or not code.strip():
        return []

    tool = _find_tool("bandit")
    tmp_path: Optional[str] = None
    findings: List[Finding] = []

    try:
        # Create temp file with .py suffix; close it so subprocess can read on Windows
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        cmd = [tool, "-f", "json", "-q", tmp_path]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        # Bandit outputs JSON to stdout; exit code 0 (no issues) or 1 (issues found)
        raw_output = res.stdout.strip()
        if not raw_output:
            return []

        data = json.loads(raw_output)
        results = data.get("results", [])

        severity_map = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}
        confidence_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}

        for item in results:
            line_no = int(item.get("line_number", 1))
            line_range = item.get("line_range", [line_no])
            end_line = int(line_range[-1]) if line_range else line_no
            sev = severity_map.get(str(item.get("issue_severity", "")).upper(), "medium")
            conf = confidence_map.get(str(item.get("issue_confidence", "")).upper(), 0.7)

            findings.append(
                Finding(
                    tool="bandit",
                    rule_id=str(item.get("test_id", "B000")),
                    category="security",
                    severity=sev,
                    message=str(item.get("issue_text", "")),
                    line=line_no,
                    end_line=end_line,
                    confidence=conf,
                    extra={
                        "test_name": item.get("test_name"),
                        "more_info": item.get("more_info"),
                    },
                )
            )
    except subprocess.TimeoutExpired:
        findings.append(
            Finding(
                tool="bandit",
                rule_id="bandit-timeout",
                category="style",
                severity="low",
                message=f"Bandit security scan timed out after {int(timeout)}s. Other checks completed normally.",
                line=1,
                end_line=1,
                confidence=0.5,
                extra={"timeout": timeout},
            )
        )
    except Exception:
        # Always return [] on unexpected tool failure, never crash
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return findings


def run_ruff(code: str, timeout: float = 10.0) -> List[Finding]:
    """Execute Ruff linter with rules F, B, E9 via temporary file.

    Returns empty list on tool failure or timeout.
    """
    if not code or not code.strip():
        return []

    tool = _find_tool("ruff")
    tmp_path: Optional[str] = None
    findings: List[Finding] = []

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        cmd = [tool, "check", "--select", "F,B,E9", "--output-format", "json", tmp_path]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        raw_output = res.stdout.strip()
        if not raw_output:
            return []

        data = json.loads(raw_output)
        if not isinstance(data, list):
            return []

        for item in data:
            rule_code = str(item.get("code", "RUFF000"))
            location = item.get("location") or {}
            end_location = item.get("end_location") or location
            start_row = int(location.get("row", 1))
            end_row = int(end_location.get("row", start_row))

            # Categorize based on rule prefix
            if rule_code.startswith("B"):
                cat = "bug"
                sev = "medium"
            elif rule_code.startswith("E9"):
                cat = "bug"
                sev = "high"
            elif rule_code.startswith("F"):
                cat = "bug"
                sev = "medium"
            elif rule_code.startswith("S"):
                cat = "security"
                sev = "high"
            else:
                cat = "style"
                sev = "low"

            findings.append(
                Finding(
                    tool="ruff",
                    rule_id=rule_code,
                    category=cat,
                    severity=sev,
                    message=str(item.get("message", "")),
                    line=start_row,
                    end_line=end_row,
                    confidence=0.85,
                    extra={"fix": item.get("fix")},
                )
            )
    except subprocess.TimeoutExpired:
        findings.append(
            Finding(
                tool="ruff",
                rule_id="ruff-timeout",
                category="style",
                severity="low",
                message=f"Ruff linter timed out after {int(timeout)}s. Other checks completed normally.",
                line=1,
                end_line=1,
                confidence=0.5,
                extra={"timeout": timeout},
            )
        )
    except Exception:
        # Always return [] on unexpected tool failure, never crash
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return findings
