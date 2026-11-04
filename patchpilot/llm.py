"""LLM enrichment layer using Gemini (google-genai SDK) with backoff retries."""

import json
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional

from patchpilot.analyzers.models import Finding

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
BACKOFF_DELAYS = (2.0, 4.0, 8.0, 12.0, 16.0)
MAX_BACKOFF_TOTAL_SECONDS = 45.0


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception indicates an HTTP 429, 500, or 503 error."""
    # Attribute checks directly on exception (e.g. google.genai.errors.APIError)
    for attr in ("code", "status_code"):
        val = getattr(exc, attr, None)
        if val in (429, 500, 503) or str(val) in ("429", "500", "503"):
            return True

    # Check underlying response object if present
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "code"):
            val = getattr(resp, attr, None)
            if val in (429, 500, 503) or str(val) in ("429", "500", "503"):
                return True

    # Check regex word match for HTTP status codes
    msg = str(exc)
    if re.search(r"\b(429|500|503)\b", msg):
        return True

    msg_lower = msg.lower()
    retry_phrases = (
        "resource_exhausted",
        "service unavailable",
        "model is overloaded",
    )
    return any(p in msg_lower for p in retry_phrases)


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


def enrich(
    code: str,
    findings: List[Finding],
    on_retry: Optional[Callable[[int, float], Any]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> List[Finding]:
    """Enrich the top 15 findings with explanations and minimal fixes via Gemini.

    On 429/500/503 errors, retries up to 5 times with exponential backoff and jitter
    (2s, 4s, 8s, 12s, 16s), capped at ~45 seconds total.
    If GEMINI_API_KEY is missing or the call still fails, returns findings unchanged
    and records the error in the first finding's extra['llm_error']. Never crashes.
    """
    if not findings:
        return findings

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        findings[0].extra["llm_error"] = "GEMINI_API_KEY environment variable is not set."
        return findings

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_GEMINI_MODEL

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

    last_error: Optional[Exception] = None
    max_retries = len(BACKOFF_DELAYS)
    total_slept = 0.0

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json",
        )

        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )

                raw_text = getattr(response, "text", "") or getattr(response, "output_text", "")
                if not raw_text and hasattr(response, "candidates") and response.candidates:
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

                last_error = None
                break

            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc) or attempt >= max_retries:
                    break

                base_delay = BACKOFF_DELAYS[attempt]
                jitter = random.uniform(0.0, 1.0)
                delay = base_delay + jitter

                remaining_cap = MAX_BACKOFF_TOTAL_SECONDS - total_slept
                if remaining_cap <= 0:
                    break
                delay = min(delay, remaining_cap)
                total_slept += delay

                if on_retry is not None:
                    try:
                        ctx = on_retry(attempt + 1, delay)
                        if hasattr(ctx, "__enter__"):
                            with ctx:
                                sleep_fn(delay)
                        else:
                            sleep_fn(delay)
                    except Exception:
                        sleep_fn(delay)
                else:
                    sleep_fn(delay)

        if last_error is not None and findings:
            findings[0].extra["llm_error"] = f"LLM enrichment failed: {last_error}"

    except Exception as exc:
        if findings:
            findings[0].extra["llm_error"] = f"LLM enrichment failed: {exc}"

    return findings
