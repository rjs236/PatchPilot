"""PatchPilot end-to-end review pipeline orchestrator."""

from typing import Any, Dict, List

from ml.inference import map_function_risks_to_lines, predict_risk
from patchpilot.analyzers.ast_analyzer import analyze_ast
from patchpilot.analyzers.external import run_bandit, run_ruff
from patchpilot.llm import enrich
from patchpilot.ranker import rank_findings


def review(
    code: str,
    filename: str = "snippet.py",
    use_llm: bool = False,
    use_ml: bool = True,
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
    code = code or ""
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
