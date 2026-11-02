"""PatchPilot end-to-end review pipeline orchestrator."""

from typing import Any, Dict, List

from patchpilot.analyzers.ast_analyzer import analyze_ast
from patchpilot.analyzers.external import run_bandit, run_ruff
from patchpilot.llm import enrich
from patchpilot.ranker import rank_findings


def review(
    code: str,
    filename: str = "snippet.py",
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Execute the multi-layered review pipeline on Python code.

    Args:
        code: The Python source code string.
        filename: Optional name of the file being reviewed.
        use_llm: Whether to invoke Gemini LLM enrichment for top findings.

    Returns:
        Dictionary with filename, line count, severity summary, and findings.
    """
    code = code or ""
    line_count = len(code.splitlines()) if code else 0

    # 1. Static Analysis Layer
    ast_findings = analyze_ast(code, filename=filename)
    bandit_findings = run_bandit(code)
    ruff_findings = run_ruff(code)

    all_findings = ast_findings + bandit_findings + ruff_findings

    # 2. Ranking & De-duplication
    ranked = rank_findings(all_findings)

    # 3. LLM Enrichment Layer
    if use_llm and ranked:
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
        "findings": [f.to_dict() for f in ranked],
    }
