"""Streamlit UI for PatchPilot automated code review."""

import os
import streamlit as st

from patchpilot.pipeline import review

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
        background-color: #374151;
        color: #e5e7eb;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("⚡ PatchPilot")
st.sidebar.caption("Future Tech 2026 • AI-Powered Automated Code Review")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuration")
use_llm = st.sidebar.checkbox(
    "Enable Gemini LLM Enrichment",
    value=True,
    help="Leverages Gemini via google-genai SDK to provide plain English explanations and automated code fixes.",
)

has_gemini_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
if has_gemini_key:
    st.sidebar.success("✓ GEMINI_API_KEY detected")
else:
    st.sidebar.warning("⚠️ No GEMINI_API_KEY set. Reviews will run via AST/Bandit/Ruff static analyzers.")

gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
st.sidebar.caption(f"LLM Model: `{gemini_model}`")

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Quick Samples")
sample_choice = st.sidebar.radio(
    "Load test scenario:",
    ["Custom Code", "Vulnerable & Buggy", "Complex & Deeply Nested", "Clean Idiomatic Code"],
)

default_code = SAMPLE_VULNERABLE
if sample_choice == "Complex & Deeply Nested":
    default_code = SAMPLE_COMPLEX
elif sample_choice == "Clean Idiomatic Code":
    default_code = SAMPLE_CLEAN
elif sample_choice == "Custom Code":
    default_code = ""

# Main Content
st.title("PatchPilot")
st.markdown(
    "Multi-layered hybrid code reviewer combining **AST semantic rules**, "
    "**Bandit security analysis**, **Ruff linting**, and **Gemini AI explanation & minimal fixes**."
)

input_tab, upload_tab = st.tabs(["📝 Paste Code", "📁 Upload .py File"])

code_to_review = default_code
filename = "snippet.py"

with input_tab:
    code_input = st.text_area(
        "Python Code",
        value=default_code,
        height=280,
        placeholder="Paste your Python source code here...",
    )
    if code_input:
        code_to_review = code_input

with upload_tab:
    uploaded_file = st.file_uploader("Upload Python script (.py)", type=["py"])
    if uploaded_file is not None:
        filename = uploaded_file.name
        code_to_review = uploaded_file.getvalue().decode("utf-8")
        st.info(f"Loaded `{filename}` ({len(code_to_review.splitlines())} lines)")

col_btn, _ = st.columns([1, 4])
with col_btn:
    run_clicked = st.button("🚀 Analyze Code", type="primary", use_container_width=True)

if run_clicked or ("review_result" in st.session_state and not default_code):
    if not code_to_review.strip():
        st.warning("Please enter or upload Python code to analyze.")
    else:
        with st.spinner("Reviewing code across static analysis, ranking, and AI layers..."):
            result = review(code=code_to_review, filename=filename, use_llm=use_llm)
            st.session_state["review_result"] = result

if "review_result" in st.session_state:
    res = st.session_state["review_result"]
    summary = res.get("summary", {})
    findings = res.get("findings", [])

    st.markdown("---")
    st.subheader("📊 Review Summary")

    # Metrics row
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

    st.markdown("<br>", unsafe_allow_html=True)

    if not findings:
        st.success("🎉 No issues detected! The submitted code passed all static and quality checks.")
    else:
        st.subheader(f"Findings ({len(findings)}) — Ordered by Severity & Risk Score")

        for idx, f in enumerate(findings, start=1):
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

            header_text = (
                f"#{idx} [{sev.upper()}] Line {line}:{end_line} • {rule_id} ({tool}) — {msg[:70]}"
            )

            with st.expander(header_text, expanded=(idx <= 3)):
                col_info1, col_info2 = st.columns([3, 1])
                with col_info1:
                    st.markdown(
                        f'<span class="{badge_class}">{sev.upper()}</span> &nbsp; '
                        f'<span class="badge-cat">{cat.upper()}</span> &nbsp; '
                        f'<code>Tool: {tool}</code> &nbsp; <code>Rule: {rule_id}</code>',
                        unsafe_allow_html=True,
                    )
                with col_info2:
                    st.markdown(f"**Risk Score:** `{score:.2f}`")

                st.markdown(f"**Issue Description:** {msg}")

                if extra.get("llm_error"):
                    st.info(f"ℹ️ LLM Notice: {extra['llm_error']}")

                if explanation:
                    st.markdown("#### 💡 AI Root-Cause & Risk Analysis")
                    st.markdown(explanation)

                if suggested_fix:
                    st.markdown("#### 🛠️ Suggested Minimal Fix")
                    st.code(suggested_fix, language="python")
