"""Unified diff parsing and line-based finding filtering for GitHub pull requests."""

import re
from typing import Dict, List, Optional, Set, Tuple

from patchpilot.analyzers.models import Finding

HUNK_HEADER_RE = re.compile(r"^@@\s+-(?:\d+)(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@")


def parse_patch_changed_lines(patch: Optional[str]) -> Set[int]:
    """Parse a unified-diff patch string and extract 1-indexed line numbers added or modified in the new file.

    Handles:
    - Multiple hunks with standard @@ -l,s +l,s @@ headers
    - Single-line hunks where the count is omitted (e.g., @@ -1 +1 @@)
    - Deletion-only hunks (which produce no new added lines)
    - "\\ No newline at end of file" comments
    """
    if not patch or not isinstance(patch, str):
        return set()

    changed_lines: Set[int] = set()
    current_new_line = 0
    in_hunk = False

    for line in patch.splitlines():
        if line.startswith("@@"):
            match = HUNK_HEADER_RE.match(line)
            if match:
                current_new_line = int(match.group(1))
                in_hunk = True
            else:
                in_hunk = False
            continue

        if not in_hunk:
            continue

        if line.startswith("+"):
            changed_lines.add(current_new_line)
            current_new_line += 1
        elif line.startswith("-"):
            # Deletion from old file: does not advance new file line counter
            continue
        elif line.startswith(" "):
            # Unchanged context line in new file
            current_new_line += 1
        elif line.startswith(r"\ "):
            # '\ No newline at end of file' indicator
            continue

    return changed_lines


def filter_findings_by_diff(
    findings: List[Finding], changed_lines: Set[int]
) -> Tuple[List[Finding], int]:
    """Filter findings so that only findings touching changed lines in the PR are retained.

    A finding is retained if its reported line or any line in its [line, end_line]
    span overlaps with changed_lines.

    Returns:
        (retained_findings, pre_existing_hidden_count)
    """
    if not changed_lines:
        return [], len(findings)

    retained: List[Finding] = []
    hidden_count = 0

    for finding in findings:
        start = finding.line
        end = finding.end_line if finding.end_line is not None else start
        # Check if any line in [start, end] is in changed_lines
        finding_lines = range(start, end + 1)
        if any(line in changed_lines for line in finding_lines):
            retained.append(finding)
        else:
            hidden_count += 1

    return retained, hidden_count


def filter_ml_risks_by_diff(
    ml_risks: Dict[int, float], changed_lines: Set[int]
) -> Dict[int, float]:
    """Retain ML line risk scores only for line numbers that fall on changed lines."""
    if not changed_lines or not ml_risks:
        return {}
    return {line: risk for line, risk in ml_risks.items() if line in changed_lines}
