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
2. **Layer 2: Multi-Factor Scoring, ML Risk Prior & Ranking Engine**
   - Ranks findings by a composite risk score:
     $$\text{Score} = \text{SeverityWeight} \times \text{Confidence} \times \text{CategoryWeight} \times (1 + \text{MLRisk})$$
   - **Trained ML Code Risk Classifier**: On production (non-test) Python functions from PyCode-Vul, PatchPilot's gradient-boosting risk model reaches ROC-AUC 0.82 [0.80-0.84] and F1 0.46 [0.41-0.51] on a cluster-isolated random split (N=2,236; 23% vulnerable; majority-class accuracy 76.9%), with recall of only 32% at the default threshold. On a held-out-repository split (N=1,052; 84 positives, mostly from two repositories) it reaches ROC-AUC 0.96 [0.94-0.98] and F1 0.73 [0.65-0.80]; this benchmark is small and imbalanced, so we treat it as supporting evidence only. Our audit found that 42% of the raw dataset were unit tests with a 74% vulnerable-label rate caused by fix-commit mining, and a trivial is_test baseline alone reached 0.70 AUC, so we removed tests, assertion counts and repository-identifying tokens from training. Labels remain noisy, and the model learns to separate vulnerable from patched code rather than to discover new vulnerabilities, so it is used only as a risk prior that re-ranks findings from Bandit, Ruff and custom AST checks.
   - Intelligent de-duplication merges alerts colliding on `(line, category)`, ensuring developers only see the highest-impact finding.
3. **Layer 3: Gemini AI Enrichment (`google-genai` SDK)**
   - Sends file context and top 15 prioritized findings to Gemini in **one single batched JSON request** (temperature 0.2).
   - Provides clear, plain English root-cause diagnosis.
   - Generates minimal, copy-pasteable production fixes.
   - **Resilient Fallback**: If offline or no API key is set, review completes without disruption.

---

## 📊 Benchmark & Empirical Evaluation

On the main benchmark, Bandit alone has higher F1 (0.768) than full PatchPilot (0.575) and higher recall; static PatchPilot reaches recall 0.905 but with a 49% false-positive rate caused mostly by maintainability rules; once the ML prior is applied with a validation-chosen threshold, full PatchPilot behaves much like the ML model alone. Note that our audit revealed 86% of Bandit's true positives came from generic constructs (asserts, bare excepts, /tmp strings) rather than the patched flaw; the benchmark measures agreement with commit-mined labels, not detection of the specific vulnerabilities. All decision thresholds were tuned on validation partitions and applied frozen to test sets without LLM calls on production (non-test) Python functions from PyCode-Vul ($N=2,236$; 23.1% positive rate).

| Setup | Precision | Recall | F1 Score [95% CI] | FPR (Clean) | ROC-AUC [95% CI] | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Bandit Alone** | **0.820** | 0.723 | **0.768 [0.737, 0.796]** | **0.048** | **0.849 [0.827, 0.869]** | 43 ms/func |
| **B. Static PatchPilot** (AST + Bandit + Ruff) | 0.355 | **0.905** | 0.510 [0.484, 0.537] | 0.492 | 0.761 [0.739, 0.783] | 55 ms/func |
| **C. Full PatchPilot** (Static + ML Prior) | 0.630 | 0.529 | 0.575 [0.537, 0.613] | 0.093 | 0.834 [0.814, 0.853] | 60 ms/func |
| **D. ML Risk Prior Alone** (Reference) | 0.608 | 0.545 | 0.575 [0.536, 0.612] | 0.105 | 0.818 [0.796, 0.840] | **5 ms/func** |

> **Key Trade-Offs & Findings**:
> - **Bandit vs. PatchPilot Performance**: Bandit alone achieves higher F1 (0.768 vs. 0.575) and lower false-positive rate (4.8% vs. 9.3%) than full PatchPilot. PatchPilot's static analyzer achieves higher recall (**90.5%** vs. 72.3%, recovering 94 missed vulnerabilities), but has a 49.2% false-positive rate under the all-category setup due to maintainability rules.
> - **Ranking Discrimination**: Incorporating the ML risk prior improves overall ranking discrimination from **0.761 [0.739, 0.783]** to **0.834 [0.814, 0.853]** ROC-AUC and increases PR-AUC from 0.486 to 0.650.
> - **Bandit Audit Limitation**: 86.1% of Bandit True Positives trigger on generic risky API presence (such as `assert`, bare `except:`, `random`, or `/tmp`) rather than the actual line fixed in the vulnerability patch commit.
> - **Exploratory Category-Restricted Analysis**: Restricting findings to security and bug categories only (excluding complexity and style) reduces Static FPR to 6.6% (F1 = 0.756) and Full PatchPilot FPR to 7.7% (F1 = 0.757).
> - **Benchmark Hygiene Check (Excluded Unparseable Functions)**: Functions with dataset truncation (AST syntax errors after dedent) correlate with vulnerable labels (26 vs. 13 on Split A). Excluding these 39 unparseable functions ($N=2,197$) leaves all conclusions unchanged: Bandit F1 is 0.789 (+0.021), Static PatchPilot recall is 90.0% (-0.5%) with 48.9% FPR, and Full PatchPilot F1 is 0.559 (-0.017) with 8.6% FPR.
> - **Supporting Repository-Holdout Split (Split B, $N=1,052$)**: Full PatchPilot achieves ROC-AUC **0.928 [0.901, 0.954]** and F1 0.469 [0.399, 0.544] with an FPR of 14.0%; Bandit alone achieves F1 0.672 [0.571, 0.757] with an FPR of 0.7%.
> - **Full Details**: Detailed methodology, syntax error audit, and replication commands are documented in [`docs/results.md`](file:///c:/Users/rajas/OneDrive/Dokumen/Hackathon/docs/results.md) and [`eval/README.md`](file:///c:/Users/rajas/OneDrive/Dokumen/Hackathon/eval/README.md).

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
Add your Google Gemini API key and optional GitHub token:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Optional: GitHub token to increase GitHub API rate limits from 60 req/hr to 5,000 req/hr
# GITHUB_TOKEN=your_personal_access_token_here
```
*(Notes: If no Gemini key is set, PatchPilot runs offline static & ML analysis without LLM enrichment. If no GitHub token is set, public PR reviews work out-of-the-box within the 60 requests/hour unauthenticated quota. Tokens are never stored, logged, or returned in errors).*

---

## 💻 Running PatchPilot

### Option A: Launch Interactive Streamlit UI
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
The interface features two dedicated workflows:
- **Tab 1: 📝 Review Code Snippet**: Live code editor with syntax highlighting, `.py` file drag-and-drop uploader, and quick preset scenarios (*Vulnerable & Buggy*, *Complex & Deeply Nested*, *Clean Idiomatic Code*).
- **Tab 2: 🐙 Review a Pull Request**: Paste any public GitHub PR URL (e.g. `https://github.com/owner/repo/pull/123`), configure LLM and ML toggles, or click **📦 Load Demo PR (Offline)** for an instant demo without network or rate limit concerns. Shows PR metadata, file-grouped findings on added/modified lines, hidden pre-existing counts, and skipped file reasons.

### Option B: Start FastAPI REST Server
```bash
uvicorn patchpilot.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

#### 1. Code Snippet Review (`POST /review`)
```bash
curl -X POST "http://localhost:8000/review" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def run(x=[]):\n    eval(x)\n",
    "filename": "vuln.py",
    "use_llm": false,
    "use_ml": true
  }'
```

#### 2. Pull Request Review (`POST /review/pr`)
```bash
curl -X POST "http://localhost:8000/review/pr" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/psf/requests/pull/1234",
    "use_llm": false,
    "use_ml": true
  }'
```

#### 3. Bundled Offline Demo PR (`GET /review/pr/demo`)
```bash
curl -X GET "http://localhost:8000/review/pr/demo"
```

---

## 🐙 GitHub Pull Request Review & Diff Filtering

PatchPilot brings the review workflow directly to public GitHub pull requests:

### Realistic Code Review Workflow
- **Unified Diff Parsing**: Analyzes GitHub patch text to identify exactly which lines were added or modified in the PR.
- **Diff-Aware Filtering**: Only findings that overlap changed lines (`start..end` line range in diff set) are surfaced in the final review. Pre-existing issues in unchanged lines are suppressed and reported as `"N pre-existing issues hidden"`.
- **Function Risk Alignment**: ML function risk scores are linked only to functions that touch modified lines.

### Safe GitHub Client & Resource Limits
- **Public GitHub REST API**: Queries PR metadata, paginates changed files, and retrieves head commit raw file contents.
- **Resource Bounds**: Reviews at most **20 changed Python files** and **2,000 changed lines** per PR. Excess files or lines are safely skipped with a transparent audit list.
- **File Filters**: Skips removed files, binary or oversized patches without diff text, and non-UTF-8 files.
- **Rate Limit Resilience**: Unauthenticated requests are limited by GitHub to 60 req/hr. PatchPilot inspects `x-ratelimit-remaining` and `x-ratelimit-reset` headers and displays the exact UTC reset timestamp. Providing an optional `GITHUB_TOKEN` raises this quota to 5,000 req/hr.
- **Token Privacy**: GitHub tokens are never logged, stored, printed, or exposed in any error message or API response.

### 100% Offline Demo Mode
A sample PR response is bundled at `patchpilot/data/demo_pr.json` featuring realistic Python vulnerabilities (SQL formatting, `subprocess(shell=True)`, hardcoded secret, bare except, and mutable default). Clicking **Load Demo PR** in Streamlit or calling `GET /review/pr/demo` runs completely offline without any GitHub or Gemini API calls.

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
