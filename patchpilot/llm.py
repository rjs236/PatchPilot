"""LLM enrichment layer using Gemini (google-genai SDK)."""

import json
import os
from typing import Any, Dict, List

from patchpilot.analyzers.models import Finding


def _extract_context(code_lines: List[str], line: int, end_line: int, context_lines: int = 3) -> str:
    """Extract context lines surrounding a finding's line range."""
    if not code_lines:
        return ""
    start_idx = max(1, line - context_lines)
    end_idx = min(len(code_lines), max(line, end_line) + context_lines)
    snippet_lines = []
    for idx in range(start_idx, end_idx + 1):
        prefix = "-> " if idx == line else "   "
        snippet_lines.append(f"{prefix}{idx:4d} | {code_lines[idx - 1]}")
    return "\n".join(snippet_lines)


def enrich(code: str, findings: List[Finding]) -> List[Finding]:
    """Enrich the top 15 findings with explanations and minimal fixes via Gemini.

    If GEMINI_API_KEY is missing or the call fails, returns findings unchanged
    and records the error in the first finding's extra['llm_error']. Never crashes.
    """
    if not findings:
        return findings

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        findings[0].extra["llm_error"] = "GEMINI_API_KEY environment variable is not set."
        return findings

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not model_name:
        model_name = "gemini-2.5-flash"

    # Select top 15 findings
    target_findings = findings[:15]
    code_lines = code.splitlines()

    findings_payload = []
    for idx, f in enumerate(target_findings):
        findings_payload.append(
            {
                "index": idx,
                "tool": f.tool,
                "rule_id": f.rule_id,
                "category": f.category,
                "severity": f.severity,
                "message": f.message,
                "line": f.line,
                "end_line": f.end_line,
                "context_code": _extract_context(code_lines, f.line, f.end_line),
            }
        )

    system_prompt = (
        "You are an expert Python security and code quality reviewer. "
        "Review the provided code and findings. For each finding, provide:\n"
        "1. 'explanation': A concise, plain English explanation of the root cause, risk, and impact.\n"
        "2. 'suggested_fix': A minimal replacement code snippet or clean pattern to fix it.\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "enrichments": [\n'
        '    {\n'
        '      "index": 0,\n'
        '      "explanation": "...",\n'
        '      "suggested_fix": "..."\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"Full Source Code:\n```python\n{code}\n```\n\n"
        f"Findings to Enrich:\n{json.dumps(findings_payload, indent=2)}"
    )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json",
        )

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=config,
        )

        raw_text = getattr(response, "text", "") or getattr(response, "output_text", "")
        if not raw_text and hasattr(response, "candidates") and response.candidates:
            # Fallback extraction from candidates
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                raw_text = "".join(part.text or "" for part in candidate.content.parts)

        parsed_data = json.loads(raw_text) if raw_text else {}
        enrichments = parsed_data.get("enrichments", [])

        # Map back to findings by index
        enrichment_map: Dict[int, Dict[str, Any]] = {
            item["index"]: item for item in enrichments if isinstance(item, dict) and "index" in item
        }

        for idx, f in enumerate(target_findings):
            if idx in enrichment_map:
                enr = enrichment_map[idx]
                f.explanation = str(enr.get("explanation", "")).strip()
                f.suggested_fix = str(enr.get("suggested_fix", "")).strip()

    except Exception as exc:
        if findings:
            findings[0].extra["llm_error"] = f"LLM enrichment failed: {exc}"

    return findings
