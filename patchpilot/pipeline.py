"""PatchPilot end-to-end review pipeline orchestrator."""

import json
from pathlib import Path
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ml.inference import map_function_risks_to_lines, predict_risk
from patchpilot.analyzers.ast_analyzer import analyze_ast
from patchpilot.analyzers.external import run_bandit, run_ruff
from patchpilot.diff_utils import filter_findings_by_diff, filter_ml_risks_by_diff
from patchpilot.github_pr import GitHubPRClient
from patchpilot.llm import enrich
from patchpilot.ranker import rank_findings

MAX_LLM_FILES_PER_PR = 5


def review(
    code: str,
    filename: str = "snippet.py",
    use_llm: bool = False,
    use_ml: bool = True,
    on_llm_retry: Optional[Callable[[int, float], Any]] = None,
) -> Dict[str, Any]:
    """Execute the multi-layered review pipeline on Python code.

    Args:
        code: The Python source code string.
        filename: Optional name of the file being reviewed.
        use_llm: Whether to invoke Gemini LLM enrichment for top findings.
        use_ml: Whether to invoke the ML function risk classifier.

    Returns:
        Dictionary with filename, line count, severity summary, findings, and high-risk functions.
    """
    code = textwrap.dedent(code or "")
    line_count = len(code.splitlines()) if code else 0

    # 1. Static Analysis Layer
    ast_findings = analyze_ast(code, filename=filename)
    bandit_findings = run_bandit(code)
    ruff_findings = run_ruff(code)

    all_findings = ast_findings + bandit_findings + ruff_findings

    # 2. ML Risk Classifier Layer
    ml_line_risks: Dict[int, float] = {}
    func_risks: Dict[int, float] = {}
    high_risk_functions: List[Dict[str, Any]] = []

    if use_ml and code.strip():
        func_risks = predict_risk(code, file_path=filename)
        ml_line_risks = map_function_risks_to_lines(code, func_risks)
        high_risk_functions = [
            {"line": line, "risk": prob}
            for line, prob in sorted(func_risks.items(), key=lambda x: -x[1])
            if prob >= 0.50
        ]

    # 3. Ranking & De-duplication (integrating ML risk weights)
    ranked = rank_findings(all_findings, ml_risks=ml_line_risks)

    # 4. LLM Enrichment Layer
    if use_llm and ranked:
        if on_llm_retry is not None:
            ranked = enrich(code, ranked, on_retry=on_llm_retry)
        else:
            ranked = enrich(code, ranked)

    # Compute severity summary
    summary = {
        "high": sum(1 for f in ranked if f.severity.lower() == "high"),
        "medium": sum(1 for f in ranked if f.severity.lower() == "medium"),
        "low": sum(1 for f in ranked if f.severity.lower() == "low"),
        "total": len(ranked),
    }

    return {
        "filename": filename,
        "lines": line_count,
        "summary": summary,
        "high_risk_functions": high_risk_functions,
        "findings": [f.to_dict() for f in ranked],
    }


def review_pull_request(
    url: str,
    use_llm: bool = False,
    use_ml: bool = True,
    token: Optional[str] = None,
    on_llm_retry: Optional[Callable[[int, float], Any]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end pull request review on changed Python files.

    Only findings touching added/modified lines in the PR diff are retained.
    Pre-existing issues on unchanged lines are filtered and tallied.

    Args:
        url: Public GitHub pull request URL.
        use_llm: Whether to invoke Gemini LLM enrichment on findings.
        use_ml: Whether to incorporate ML function risk priors.
        token: Optional GitHub API token.

    Returns:
        Structured pull request review dictionary.
    """
    client = GitHubPRClient(token=token)
    metadata, eligible_files, skipped_files = client.fetch_pr(url)

    per_file_results: List[Dict[str, Any]] = []
    total_hidden_count = 0
    llm_files_enriched = 0

    for file_obj in eligible_files:
        code = file_obj.content or ""
        changed_lines = file_obj.changed_lines

        # 1. Static Analysis Layer
        ast_findings = analyze_ast(code, filename=file_obj.filename)
        bandit_findings = run_bandit(code)
        ruff_findings = run_ruff(code)
        all_findings = ast_findings + bandit_findings + ruff_findings

        # 2. ML Risk Prior (kept only for lines in changed diff)
        ml_line_risks: Dict[int, float] = {}
        if use_ml and code.strip():
            raw_func_risks = predict_risk(code, file_path=file_obj.filename)
            raw_line_risks = map_function_risks_to_lines(code, raw_func_risks)
            ml_line_risks = filter_ml_risks_by_diff(raw_line_risks, changed_lines)

        # 3. Ranking & De-duplication
        ranked = rank_findings(all_findings, ml_risks=ml_line_risks)

        # 4. Diff Filtering: Retain only findings touching changed lines
        diff_findings, hidden_count = filter_findings_by_diff(ranked, changed_lines)
        total_hidden_count += hidden_count

        # 5. LLM Enrichment Layer (capped per PR for performance)
        if use_llm and diff_findings and llm_files_enriched < MAX_LLM_FILES_PER_PR:
            if on_llm_retry is not None:
                diff_findings = enrich(code, diff_findings, on_retry=on_llm_retry)
            else:
                diff_findings = enrich(code, diff_findings)
            llm_files_enriched += 1

        file_summary = {
            "high": sum(1 for f in diff_findings if f.severity.lower() == "high"),
            "medium": sum(1 for f in diff_findings if f.severity.lower() == "medium"),
            "low": sum(1 for f in diff_findings if f.severity.lower() == "low"),
            "total": len(diff_findings),
        }

        per_file_results.append(
            {
                "filename": file_obj.filename,
                "changed_lines": sorted(list(changed_lines)),
                "summary": file_summary,
                "findings": [f.to_dict() for f in diff_findings],
            }
        )

    # Compute aggregate PR summary
    overall_summary = {
        "high": sum(f["summary"]["high"] for f in per_file_results),
        "medium": sum(f["summary"]["medium"] for f in per_file_results),
        "low": sum(f["summary"]["low"] for f in per_file_results),
        "total": sum(f["summary"]["total"] for f in per_file_results),
    }

    return {
        "pr_metadata": {
            "owner": metadata.owner,
            "repo": metadata.repo,
            "pull_number": metadata.pull_number,
            "title": metadata.title,
            "author": metadata.author,
            "base_branch": metadata.base_branch,
            "head_branch": metadata.head_branch,
            "head_sha": metadata.head_sha,
            "html_url": metadata.html_url,
        },
        "files": per_file_results,
        "summary": overall_summary,
        "skipped_files": skipped_files,
        "pre_existing_hidden_count": total_hidden_count,
    }


def load_demo_pr() -> Dict[str, Any]:
    """Load the bundled offline sample PR review data."""
    demo_path = Path(__file__).parent / "data" / "demo_pr.json"
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo PR file not found at: {demo_path}")
    with open(demo_path, "r", encoding="utf-8") as f:
        return json.load(f)
