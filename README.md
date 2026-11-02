# ⚡ PatchPilot: AI-Powered Automated Code Review

> **Neural Storm 2026 Hackathon** | **Track:** Future Tech | **Problem FT-05:** AI-Powered Automated Code Review

PatchPilot is a next-generation, hybrid code reviewer for Python designed to eliminate code smells, subtle logic bugs, severe security vulnerabilities, and excessive architectural complexity before code merges.

---

## 🎯 The Problem
Traditional code reviews present a persistent dilemma for modern engineering teams:
- **Human reviewers are overburdened**, often missing subtle runtime bugs, security holes, and complexity drift.
- **Pure Linters produce high false-positive noise** without explanation or remediation, leading to developer fatigue.
- **Pure LLM reviewers hallucinate syntax rules**, lack deterministic reliability, and are too costly to run indiscriminately across full repositories.

## 💡 The Solution: 3-Layer Hybrid Architecture
PatchPilot resolves this by fusing deterministic static analysis with intelligent ranking and targeted LLM synthesis:

1. **Layer 1: Static Analysis Engine**
   - **Custom Python AST Rules**: Identifies mutable default arguments, bare `except:` blocks, hazardous dynamic calls (`eval`/`exec`), loose `== None` identity comparisons, cyclomatic complexity (>10 medium, >15 high), deep nesting (>4 levels), long functions (>50 lines), excessive parameter lists (>6 args), and syntax errors.
   - **Bandit**: Scans for known Python vulnerability patterns (injection, insecure deserialization, unsafe crypto).
   - **Ruff**: Fast static linter checking AST correctness, pyflakes (`F`), flake8-bugbear (`B`), and syntax/runtime anomalies (`E9`).
2. **Layer 2: Multi-Factor Scoring & Ranking Engine**
   - Ranks findings by a composite risk score:
     $$\text{Score} = \text{SeverityWeight} \times \text{Confidence} \times \text{CategoryWeight} \times (1 + \text{MLRisk})$$
   - Intelligent de-duplication merges alerts colliding on `(line, category)`, ensuring developers only see the highest-impact finding.
3. **Layer 3: Gemini AI Enrichment (`google-genai` SDK)**
   - Sends file context and top 15 prioritized findings to Gemini in **one single batched JSON request** (temperature 0.2).
   - Provides clear, plain English root-cause diagnosis.
   - Generates minimal, copy-pasteable production fixes.
   - **Resilient Fallback**: If offline or no API key is set, review completes without disruption.

---

## 📁 Repository Structure

```
patchpilot/
├── patchpilot/               # Core Python package
│   ├── __init__.py
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── models.py         # Finding dataclass & serialization
│   │   ├── ast_analyzer.py   # Custom Python AST visitor
│   │   └── external.py       # Bandit & Ruff subprocess runners
│   ├── ranker.py             # Weighted scoring & de-duplication
│   ├── pipeline.py           # End-to-end review orchestrator
│   ├── llm.py                # Gemini enrichment layer (google-genai)
│   └── api.py                # FastAPI REST endpoints
├── app/
│   └── streamlit_app.py      # Interactive Streamlit dashboard
├── tests/                    # Pytest test suite (100% offline, mocked)
├── ml/                       # ML layer roadmap & training models (Phase 2)
├── docs/                     # Documentation & AI tools disclosure
│   ├── AI_TOOLS.md           # Mandatory AI tools disclosure log
│   └── ARCHITECTURE.md       # Architectural deep-dive
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Testing & development dependencies
├── .env.example              # Environment template
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Git

### 2. Environment Setup
```bash
# Clone and enter directory
cd patchpilot

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
*(Note: If no key is set, PatchPilot gracefully falls back to deterministic static analysis with zero runtime errors).*

---

## 💻 Running PatchPilot

### Option A: Launch Interactive Streamlit UI
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
Features:
- Live code editor with syntax highlighting
- Python file drag-and-drop uploader (`.py`)
- Preset sample scenarios: *Vulnerable & Buggy*, *Complex & Deeply Nested*, *Clean Idiomatic Code*
- Real-time summary metric cards (High, Medium, Low, Total)
- Expandable cards with severity badges, AI explanations, and suggested code diffs

### Option B: Start FastAPI REST Server
```bash
uvicorn patchpilot.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Example API Request:
```bash
curl -X POST "http://localhost:8000/review" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def run(x=[]):\n    eval(x)\n",
    "filename": "vuln.py",
    "use_llm": false
  }'
```

#### Example API Response:
```json
{
  "filename": "vuln.py",
  "lines": 2,
  "summary": {
    "high": 1,
    "medium": 1,
    "low": 0,
    "total": 2
  },
  "findings": [
    {
      "tool": "ast",
      "rule_id": "eval-exec",
      "category": "security",
      "severity": "high",
      "message": "Dynamic code execution via 'eval()' is hazardous and allows arbitrary code execution.",
      "line": 2,
      "end_line": 2,
      "confidence": 0.95,
      "score": 4.275,
      "explanation": "eval() evaluates arbitrary strings, enabling arbitrary remote code execution.",
      "suggested_fix": "# Avoid eval; use ast.literal_eval for safe literal parsing"
    }
  ]
}
```

---

## 🧪 Running Automated Tests
Run the comprehensive Pytest suite:
```bash
pytest -v
```
All tests are 100% network-isolated:
- Unit tests for all AST checks (mutable defaults, bare excepts, eval/exec, None comparison, complexity, deep nesting, long functions, too many args, syntax error recovery)
- Ranker scoring formula, de-duplication, and order verification
- External tool runner mocks and fallback validation
- FastAPI TestClient endpoints (`/health`, `/review`, 200k char bounds)
- Gemini mock verification and graceful offline handling

---

## 🤖 AI Disclosure
In adherence to the Neural Storm 2026 hackathon requirements, all AI assistance is logged in [`docs/AI_TOOLS.md`](file:///c:/Users/rajas/OneDrive/Dokumen/Hackathon/docs/AI_TOOLS.md).
- **Antigravity IDE Agent**: Used for project scaffolding, architectural coordination, and unit test generation.
- **Google Gemini (google-genai SDK)**: Used at runtime for code review explanation and fix generation.
