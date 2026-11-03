"""Scoring, ranking, and de-duplication engine for findings."""

from typing import Dict, List, Optional

from patchpilot.analyzers.models import Finding

SEVERITY_WEIGHTS: Dict[str, float] = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}

CATEGORY_WEIGHTS: Dict[str, float] = {
    "security": 1.5,
    "bug": 1.2,
    "complexity": 1.0,
    "style": 0.5,
}


def compute_finding_score(finding: Finding, ml_risk: float = 0.0) -> float:
    """Calculate normalized risk score for a single finding."""
    sev_w = SEVERITY_WEIGHTS.get(finding.severity.lower(), 1.0)
    cat_w = CATEGORY_WEIGHTS.get(finding.category.lower(), 1.0)
    conf = max(0.0, min(1.0, finding.confidence if finding.confidence is not None else 1.0))
    risk_factor = 1.0 + max(0.0, ml_risk)
    return round(sev_w * conf * cat_w * risk_factor, 3)


def rank_findings(
    findings: List[Finding],
    ml_risks: Optional[Dict[int, float]] = None,
) -> List[Finding]:
    """Score, de-duplicate by (line, category) keeping highest score, and sort descending."""
    if not findings:
        return []

    risk_map = ml_risks or {}

    # Score all findings
    scored_findings: List[Finding] = []
    for f in findings:
        line_risk = risk_map.get(f.line, 0.0)
        f.score = compute_finding_score(f, line_risk)
        if line_risk > 0:
            f.extra["ml_risk"] = round(line_risk, 3)
        scored_findings.append(f)

    # De-duplicate by (line, category), keeping the highest score
    deduped: Dict[tuple[int, str], Finding] = {}
    for f in scored_findings:
        key = (f.line, f.category.lower())
        if key not in deduped:
            deduped[key] = f
        else:
            if f.score > deduped[key].score:
                deduped[key] = f
            elif f.score == deduped[key].score and f.confidence > deduped[key].confidence:
                deduped[key] = f

    # Sort descending by score; secondary sort by line number ascending
    sorted_findings = sorted(
        deduped.values(),
        key=lambda item: (-item.score, item.line),
    )

    return sorted_findings
