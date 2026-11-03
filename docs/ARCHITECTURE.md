# PatchPilot Architecture

PatchPilot is designed as a **3-tier hybrid code reviewer** for Python, providing high precision through deterministic static checks and actionable insight through LLM enrichment.

```
                           +----------------------+
                           |   Input Source Code  |
                           |  (Pasted / Upload)   |
                           +----------+-----------+
                                      |
         +----------------------------+----------------------------+
         |                            |                            |
         v                            v                            v
+------------------+        +--------------------+       +--------------------+
|   Custom AST     |        |   Bandit Runner    |       |   Ruff Runner      |
|  Analyzer Engine |        | (Security Scanner) |       | (Rules F, B, E9)   |
+--------+---------+        +---------+----------+       +---------+----------+
         |                            |                            |
         +----------------------------+----------------------------+
                                      |
                                      v
                        +---------------------------+
                        |   All Raw Code Findings   |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |  Risk Scoring & Ranker    |
                        | - Severity Weights (3/2/1)|
                        | - Category Multipliers    |
                        | - Confidence Normalization|
                        | - Line Deduplication      |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |  Gemini AI Enrichment     |
                        |  (google-genai SDK)       |
                        | - Single Batched Request  |
                        | - Top-15 Findings Context |
                        | - Plain English Diagnosis |
                        | - Minimal Verified Fix    |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |  Clean JSON API / Web UI  |
                        | (FastAPI & Streamlit App) |
                        +---------------------------+
```

## Tier 1: Static Analysis
- **Custom Python AST Visitor (`patchpilot.analyzers.ast_analyzer`)**:
  - Mutable default argument anti-pattern (`def f(x=[])`)
  - Bare exception handling (`except:` and `except BaseException:`)
  - Hazardous dynamic execution (`eval()`, `exec()`)
  - Loose `None` identity comparisons (`== None`, `!= None`)
  - Cyclomatic complexity tracking (McCabe metric > 10 medium, > 15 high)
  - Deep nesting depth detection (> 4 levels)
  - Function length constraints (> 50 lines)
  - Argument count bounds (> 6 arguments)
  - Syntax error resilient recovery (reports high-severity issue without crashing)
- **Bandit & Ruff Subprocess Runners (`patchpilot.analyzers.external`)**:
  - Sandboxed execution on temporary files with guaranteed cleanup.
  - JSON output parsing with structured fallback on tool absence.

## Tier 2: Ranking & De-duplication Engine (`patchpilot.ranker`)
- Multi-factor risk formula:
  $$\text{Score} = \text{SeverityWeight} \times \text{Confidence} \times \text{CategoryWeight} \times (1 + \text{MLRisk})$$
- De-duplicates redundant alerts on the same `(line, category)`, preserving the highest scoring alert.
- Sorts findings descending by severity score.

## Tier 3: LLM Enrichment (`patchpilot.llm`)
- Google GenAI SDK (`google-genai`):
  - Sends full file + contextual snippets in a single structured prompt.
  - Generates plain English root-cause diagnoses and minimal code fixes.
  - Non-blocking fallback preserves all findings if offline or API key is absent.

## Tier 4: User Interface Presentation (`app/streamlit_app`)
- **Default View**: **Security** and **Bug** findings are displayed expanded by default with root-cause explanations and remediations.
- **Collapsed Maintainability Section**: Heuristic **Complexity** and **Style** findings are placed into a collapsed expander titled `🛠️ Maintainability Issues (N: X Complexity, Y Style)` to mitigate developer fatigue from architectural complexity rules.
- **Diff-Filtered PR Review**: For GitHub pull requests, unchanged lines are suppressed with a count banner showing hidden pre-existing findings.

