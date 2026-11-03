"""Streamlit UI for PatchPilot automated code review and GitHub pull request review."""

import os
import streamlit as st

from patchpilot.github_pr import (
    GitHubInvalidURLError,
    GitHubNetworkError,
    GitHubPRNotFoundError,
    GitHubRateLimitError,
    GitHubPRError,
)
from patchpilot.pipeline import load_demo_pr, review, review_pull_request

# Sample codes for instant demo testing
SAMPLE_VULNERABLE = '''import os

# Bug: mutable default argument
def process_items(items=[], extra=None):
    if extra == None:  # Style: == None comparison
        extra = {}
    items.append(extra)
    return items

def execute_user_query(query_str):
    try:
        # Security: eval is hazardous
        result = eval(query_str)
        return result
    except:  # Bug: bare except
        print("Error executing query")
        return None
'''

SAMPLE_COMPLEX = '''def calculate_route_metrics(nodes, weights, threshold, flags, mode, fallback_rate, max_iter):
    # Too many args (>6) and deeply nested logic
    total = 0
    if nodes:
        for node in nodes:
            if node.is_valid():
                for w in weights:
                    if w > threshold:
                        while total < max_iter:
                            if mode == "fast" and flags:
                                total += w * fallback_rate
                            else:
                                total += w
    return total
'''

SAMPLE_CLEAN = '''from typing import Any, Dict, List, Optional


def calculate_average(numbers: List[float]) -> Optional[float]:
    """Calculate the arithmetic mean of a non-empty sequence."""
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def fetch_user_data(user_id: int, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Retrieve user payload safely with immutable default handling."""
    opts = options if options is not None else {}
    return {"id": user_id, "options": opts}
'''

st.set_page_config(
    page_title="PatchPilot | AI Code Reviewer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e222d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2e3440;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .badge-high {
        background-color: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-medium {
        background-color: #f59e0b;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #3b82f6;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-cat {
        background-color: #4b5563;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.title("⚡ PatchPilot Settings")
st.sidebar.markdown("Configure analysis, ML risk prior, and AI review layers.")

use_ml = st.sidebar.checkbox(
    "Enable ML Risk Prior",
    value=True,
    help="Scores functions using trained HistGradientBoosting model trained on vulnerability datasets.",
)

use_llm = st.sidebar.checkbox(
    "Enable Gemini AI Enrichment",
    value=False,
    help="Calls Google Gemini to explain findings in plain English and propose minimal fixes.",
)

if use_llm:
    api_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    if api_key_set:
        st.sidebar.success("GEMINI_API_KEY detected in environment.")
    else:
        st.sidebar.warning(
            "GEMINI_API_KEY not found in environment. Reviews will fall back to static analysis."
        )

# Optional GitHub Token input in sidebar (never printed or logged)
gh_token_input = st.sidebar.text_input(
    "GitHub Token (Optional)",
    type="password",
    help="Optional personal access token for higher GitHub API rate limits (5,000/hr vs 60/hr).",
)
effective_gh_token = gh_token_input.strip() if gh_token_input else None

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**PatchPilot Architecture:**\n"
    "- Layer 1: Python AST + Bandit + Ruff\n"
    "- Layer 2: Multi-Factor Scoring & ML Risk Prior\n"
    "- Layer 3: Google Gemini LLM Explanations\n"
)

# Main Application Tabs
tab_snippet, tab_pr = st.tabs(["📝 Review Code Snippet", "🐙 Review a Pull Request"])

# ==============================================================================
# TAB 1: Review Code Snippet
# ==============================================================================
with tab_snippet:
    st.subheader("Interactive Code Reviewer")
    st.markdown(
        "Analyze standalone Python snippets or uploaded files across deterministic AST checks, "
        "Bandit, Ruff, and ML risk scoring."
    )

    sample_choice = st.selectbox(
        "Load Preset Scenario",
        ["Custom Code", "Vulnerable & Buggy", "Complex & Deeply Nested", "Clean Idiomatic Code"],
        key="snippet_preset",
    )

    default_code = SAMPLE_VULNERABLE
    if sample_choice == "Complex & Deeply Nested":
        default_code = SAMPLE_COMPLEX
    elif sample_choice == "Clean Idiomatic Code":
        default_code = SAMPLE_CLEAN
    elif sample_choice == "Custom Code":
        default_code = ""

    input_subtab, upload_subtab = st.tabs(["✏️ Edit Code", "📁 Upload File"])
    code_to_review = default_code
    filename = "snippet.py"

    with input_subtab:
        code_input = st.text_area(
            "Python Code",
            value=default_code,
            height=260,
            placeholder="Paste your Python source code here...",
            key="snippet_text",
        )
        if code_input:
            code_to_review = code_input

    with upload_subtab:
        uploaded_file = st.file_uploader("Upload Python script (.py)", type=["py"], key="snippet_upload")
        if uploaded_file is not None:
            filename = uploaded_file.name
            code_to_review = uploaded_file.getvalue().decode("utf-8")
            st.info(f"Loaded `{filename}` ({len(code_to_review.splitlines())} lines)")

    if st.button("🚀 Analyze Snippet", type="primary", key="btn_run_snippet"):
        if not code_to_review.strip():
            st.warning("Please enter or upload Python code to analyze.")
        else:
            with st.spinner("Analyzing code across AST, Bandit, Ruff, and ML layers..."):
                res = review(code=code_to_review, filename=filename, use_llm=use_llm, use_ml=use_ml)
                st.session_state["snippet_result"] = res

    if "snippet_result" in st.session_state:
        res = st.session_state["snippet_result"]
        summary = res.get("summary", {})
        findings = res.get("findings", [])
        high_risk_funcs = res.get("high_risk_functions", [])

        st.markdown("---")
        st.subheader("📊 Snippet Review Summary")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div>Total Findings</div><div class="metric-val">{summary.get("total", 0)}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div style="color:#ef4444;">High Severity</div><div class="metric-val" style="color:#ef4444;">{summary.get("high", 0)}</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div style="color:#f59e0b;">Medium Severity</div><div class="metric-val" style="color:#f59e0b;">{summary.get("medium", 0)}</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><div style="color:#3b82f6;">Low Severity</div><div class="metric-val" style="color:#3b82f6;">{summary.get("low", 0)}</div></div>',
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f'<div class="metric-card"><div>Lines of Code</div><div class="metric-val">{res.get("lines", 0)}</div></div>',
                unsafe_allow_html=True,
            )

        if high_risk_funcs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning(f"⚠️ **ML Risk Alert:** {len(high_risk_funcs)} function(s) flagged with vulnerability risk >= 50%:")
            for hf in high_risk_funcs:
                st.markdown(f"- Function at Line {hf['line']}: Predicted Risk **{hf['risk'] * 100:.1f}%**")

        st.markdown("<br>", unsafe_allow_html=True)
        if not findings:
            st.success("🎉 No issues detected! The submitted code passed all static and quality checks.")
        else:
            st.subheader(f"Findings ({len(findings)})")
            for idx, f in enumerate(findings, start=1):
                sev = f.get("severity", "low").lower()
                badge_class = f"badge-{sev}"
                score = f.get("score", 0.0)
                line = f.get("line", 1)
                end_line = f.get("end_line", line)
                tool = f.get("tool", "ast")
                rule_id = f.get("rule_id", "RULE")
                msg = f.get("message", "")
                explanation = f.get("explanation", "")
                suggested_fix = f.get("suggested_fix", "")

                header_text = f"#{idx} [{sev.upper()}] Line {line}:{end_line} • {rule_id} ({tool}) — {msg[:70]}"
                with st.expander(header_text, expanded=(idx <= 3)):
                    st.markdown(
                        f'<span class="{badge_class}">{sev.upper()}</span> &nbsp; '
                        f'<code>Tool: {tool}</code> &nbsp; <code>Rule: {rule_id}</code> &nbsp; '
                        f'**Score:** `{score:.2f}`',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Description:** {msg}")
                    if explanation:
                        st.markdown("#### 💡 AI Root-Cause Analysis")
                        st.markdown(explanation)
                    if suggested_fix:
                        st.markdown("#### 🛠️ Suggested Fix")
                        st.code(suggested_fix, language="python")

# ==============================================================================
# TAB 2: Review a Pull Request
# ==============================================================================
with tab_pr:
    st.subheader("GitHub Pull Request Code Review")
    st.markdown(
        "Paste a public GitHub Pull Request URL. PatchPilot fetches changed `.py` files, "
        "runs the full analysis pipeline, and reports **only findings on added or modified lines**."
    )

    pr_url_input = st.text_input(
        "GitHub Pull Request URL",
        placeholder="https://github.com/owner/repo/pull/123",
        key="pr_url_field",
    )

    col_btn_pr, col_btn_demo, _ = st.columns([1.5, 1.8, 3.5])
    with col_btn_pr:
        run_pr_clicked = st.button("🚀 Review Pull Request", type="primary", use_container_width=True)
    with col_btn_demo:
        load_demo_clicked = st.button("📦 Load Demo PR (Offline)", type="secondary", use_container_width=True)

    if load_demo_clicked:
        try:
            demo_data = load_demo_pr()
            st.session_state["pr_result"] = demo_data
            st.session_state["pr_is_demo"] = True
            st.success("Loaded bundled sample pull request: `octocat/demo-repo#42` (100% offline).")
        except Exception as exc:
            st.error(f"Failed to load demo PR: {exc}")

    if run_pr_clicked:
        if not pr_url_input.strip():
            st.warning("Please provide a valid GitHub Pull Request URL.")
        else:
            with st.spinner("Fetching pull request diff and analyzing changed lines..."):
                try:
                    pr_result = review_pull_request(
                        url=pr_url_input.strip(),
                        use_llm=use_llm,
                        use_ml=use_ml,
                        token=effective_gh_token,
                    )
                    st.session_state["pr_result"] = pr_result
                    st.session_state["pr_is_demo"] = False
                except GitHubInvalidURLError as exc:
                    st.error(f"❌ Invalid URL: {exc}")
                except GitHubPRNotFoundError as exc:
                    st.error(f"🔍 Not Found: {exc}")
                except GitHubRateLimitError as exc:
                    st.error(f"⏳ Rate Limited: {exc}")
                except (GitHubNetworkError, GitHubPRError) as exc:
                    st.error(f"🌐 GitHub Connection Error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error reviewing pull request: {exc}")

    # Display PR Results
    if "pr_result" in st.session_state:
        pr_res = st.session_state["pr_result"]
        pr_meta = pr_res.get("pr_metadata", {})
        overall_summary = pr_res.get("summary", {})
        files_results = pr_res.get("files", [])
        skipped_files = pr_res.get("skipped_files", [])
        hidden_count = pr_res.get("pre_existing_hidden_count", 0)
        is_demo = st.session_state.get("pr_is_demo", False)

        st.markdown("---")
        if is_demo:
            st.info("ℹ️ Currently viewing **Offline Demo Mode** (sample pull request with deliberate security flaws).")

        # Pull Request Header Information
        st.markdown(f"### 🔀 [{pr_meta.get('title', 'Pull Request')}]({pr_meta.get('html_url', '#')})")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"**Author:** `{pr_meta.get('author', 'unknown')}`")
            st.markdown(f"**PR Number:** `#{pr_meta.get('pull_number', '-')}`")
        with col_m2:
            st.markdown(f"**Base Branch:** `{pr_meta.get('base_branch', 'main')}`")
            st.markdown(f"**Head Branch:** `{pr_meta.get('head_branch', 'feature')}`")
        with col_m3:
            st.markdown(f"**Head Commit SHA:** `{pr_meta.get('head_sha', '')[:10]}`")

        # Summary Metric Cards
        st.markdown("<br>", unsafe_allow_html=True)
        pm1, pm2, pm3, pm4 = st.columns(4)
        with pm1:
            st.markdown(
                f'<div class="metric-card"><div>Total Findings (Changed Lines)</div><div class="metric-val">{overall_summary.get("total", 0)}</div></div>',
                unsafe_allow_html=True,
            )
        with pm2:
            st.markdown(
                f'<div class="metric-card"><div style="color:#ef4444;">High Severity</div><div class="metric-val" style="color:#ef4444;">{overall_summary.get("high", 0)}</div></div>',
                unsafe_allow_html=True,
            )
        with pm3:
            st.markdown(
                f'<div class="metric-card"><div style="color:#f59e0b;">Medium Severity</div><div class="metric-val" style="color:#f59e0b;">{overall_summary.get("medium", 0)}</div></div>',
                unsafe_allow_html=True,
            )
        with pm4:
            st.markdown(
                f'<div class="metric-card"><div style="color:#3b82f6;">Low Severity</div><div class="metric-val" style="color:#3b82f6;">{overall_summary.get("low", 0)}</div></div>',
                unsafe_allow_html=True,
            )

        # Pre-existing issues notification banner
        if hidden_count > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                f"🛡️ **Pre-existing Issues Hidden**: **{hidden_count}** finding(s) on unchanged lines in these files "
                "were filtered out to keep review focused strictly on PR diff changes."
            )

        # Skipped files expander
        if skipped_files:
            with st.expander(f"⚠️ Skipped / Omitted Files ({len(skipped_files)})", expanded=False):
                for sf in skipped_files:
                    st.markdown(f"- `{sf.get('filename')}`: {sf.get('reason')}")

        st.markdown("---")
        st.subheader(f"📂 Reviewed Changed Files ({len(files_results)})")

        if not files_results:
            st.info("No changed Python files found in this pull request to review.")

        for f_idx, file_res in enumerate(files_results, start=1):
            f_name = file_res.get("filename", "")
            f_summary = file_res.get("summary", {})
            f_changed = file_res.get("changed_lines", [])
            f_findings = file_res.get("findings", [])

            st.markdown(f"#### 📄 `{f_name}`")
            st.markdown(
                f"**Changed Lines:** `{len(f_changed)} lines` (Lines: {', '.join(str(l) for l in f_changed[:15])}"
                f"{'...' if len(f_changed) > 15 else ''}) • "
                f"**Findings on Changed Lines:** `{f_summary.get('total', 0)}` "
                f"*(High: {f_summary.get('high', 0)}, Medium: {f_summary.get('medium', 0)}, Low: {f_summary.get('low', 0)})*"
            )

            if not f_findings:
                st.success(f"✅ No issues detected on changed lines in `{f_name}`.")
            else:
                for idx, f in enumerate(f_findings, start=1):
                    sev = f.get("severity", "low").lower()
                    badge_class = f"badge-{sev}"
                    cat = f.get("category", "general")
                    score = f.get("score", 0.0)
                    line = f.get("line", 1)
                    end_line = f.get("end_line", line)
                    tool = f.get("tool", "ast")
                    rule_id = f.get("rule_id", "RULE")
                    msg = f.get("message", "")
                    explanation = f.get("explanation", "")
                    suggested_fix = f.get("suggested_fix", "")
                    extra = f.get("extra", {})
                    ml_risk = extra.get("ml_risk")

                    ml_badge_str = f" [ML Risk: {int(ml_risk * 100)}%]" if ml_risk is not None else ""
                    header_text = (
                        f"#{idx} [{sev.upper()}]{ml_badge_str} Line {line}:{end_line} • {rule_id} ({tool}) — {msg[:75]}"
                    )

                    with st.expander(header_text, expanded=(idx <= 2)):
                        col_info1, col_info2 = st.columns([3, 1])
                        with col_info1:
                            ml_badge_html = (
                                f'<span style="background-color:#7c3aed;color:white;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:bold;">ML RISK: {ml_risk * 100:.1f}%</span> &nbsp; '
                                if ml_risk is not None
                                else ""
                            )
                            st.markdown(
                                f'<span class="{badge_class}">{sev.upper()}</span> &nbsp; '
                                f'<span class="badge-cat">{cat.upper()}</span> &nbsp; '
                                f'{ml_badge_html}'
                                f'<code>Tool: {tool}</code> &nbsp; <code>Rule: {rule_id}</code>',
                                unsafe_allow_html=True,
                            )
                        with col_info2:
                            st.markdown(f"**Risk Score:** `{score:.2f}`")

                        st.markdown(f"**Issue Description:** {msg}")

                        if explanation:
                            st.markdown("#### 💡 AI Root-Cause & Risk Analysis")
                            st.markdown(explanation)

                        if suggested_fix:
                            st.markdown("#### 🛠️ Suggested Fix")
                            st.code(suggested_fix, language="python")

            st.markdown("<br>", unsafe_allow_html=True)
