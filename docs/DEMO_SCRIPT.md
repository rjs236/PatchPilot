# PatchPilot Demo Video Script (2:45)
**Track:** Future Tech (FT-05: AI-Powered Automated Code Review)  
**Target Duration:** 2 minutes 30 seconds to 3 minutes (Target: 2:45)  
**Tools Displayed:** PatchPilot Web UI (Streamlit), Gemini Explanation, GitHub PR Diff Analyzer.

---

### [0:00 – 0:20] 1. The Problem & Why It Matters (20 seconds)
**Screen:** Show PatchPilot title header in Streamlit UI (`http://localhost:8501`) with the subtitle and the sidebar showing the active system status.

**Voiceover:**
> "Reviewing code changes is hard, noisy, and slow. Static analysis tools like Bandit and Ruff catch critical security flaws, but they also deluge developers with hundreds of style and maintainability alerts. Meanwhile, pure LLM reviewers suffer from hallucinations and inconsistent rule enforcement.  
> We built PatchPilot: a three-tier automated code review assistant that unifies fast deterministic AST and linter rules, a machine-learning vulnerability prior, and grounded Gemini 2.5 explanations into a single actionable interface."

---

### [0:20 – 1:00] 2. Live Snippet Review: Security Flaw, Scoring, and Gemini Fix (40 seconds)
**Screen Action:**
1. Click the **"📝 Review Code Snippet"** tab.
2. Select the sample snippet or paste the following vulnerable Python code into the code box:

```python
import subprocess
import os

def render_user_avatar(username, image_data):
    # CRITICAL: Insecure command execution with user-controlled input
    filename = f"/tmp/{username}.png"
    with open(filename, "wb") as f:
        f.write(image_data)
    
    # Vulnerable to shell injection
    cmd = f"convert {filename} -resize 128x128 /tmp/thumb_{username}.png"
    subprocess.call(cmd, shell=True)
    return f"/tmp/thumb_{username}.png"
```

3. Ensure **"Enable Gemini AI Enrichment"** is checked in the sidebar.
4. Click the **"🚀 Analyze Code Snippet"** button.

**Voiceover:**
> "Let's review a real security flaw: image resizing code with an unsanitized command string passed to `subprocess.call` with `shell=True`.  
> Within milliseconds, PatchPilot's first two layers fire. Bandit flags rule `B602` for command injection risk, and our AST analyzer detects string interpolation into an active shell.  
> Next, Gemini 2.5 steps in—grounded by the static finding—to synthesize an explanation without hallucinating. It explains the exact exploit path, classifies the vulnerability, and generates an immediate, drop-in replacement using a sanitized argument list and `shell=False`."

---

### [1:00 – 1:40] 3. Offline Demo PR Review: Diff-Aware Changed-Line Filtering (40 seconds)
**Screen Action:**
1. Switch to the **"🐙 Review a Pull Request"** tab.
2. Click the button: **"📦 Load Demo PR (Sample Data)"**.
3. Scroll down to inspect the results:
   - Point out the **PR Summary Metric Cards**: Total Findings on Changed Lines: **5** (3 High, 1 Medium, 1 Low).
   - Point out the **Pre-existing Issues Hidden** notification banner: **4 findings** on unchanged lines in these files were filtered out.
   - Note the **Skipped Files** container (2 files: `README.md` non-Python, `tests/test_legacy.py` deleted).
   - Point out the two reviewed changed files:
     * `auth_service.py` (7 changed lines): 3 findings open by default under Security & Bug—Line 12 [HIGH] `B105` (hardcoded JWT secret), Line 25 [MEDIUM] `bare-except`, Line 31 [LOW] `mutable-default-args`.
     * `backup_exporter.py` (5 changed lines): 2 findings open by default under Security & Bug—Line 19 [HIGH] `B608` (SQL injection via string formatting), Line 35 [HIGH] `B602` (subprocess call with `shell=True`).

**Voiceover:**
> "Now let's examine the pull request review workflow. Developers dread linters that complain about legacy code they didn't touch. PatchPilot solves this with diff-aware filtering.  
> Here we load our offline sample PR—`octocat/demo-repo#42`—with two modified files: `auth_service.py` and `backup_exporter.py`.  
> Across both files, PatchPilot parses the Git unified diff, isolates the added lines, and surfaces exactly the five new flaws: a hardcoded JWT secret on line 12, a bare except on line 25, a mutable default list on line 31, a SQL injection on line 19, and a command injection on line 35. Meanwhile, 4 pre-existing legacy findings on untouched lines are cleanly tucked away behind a notification banner. No alert fatigue, no wasted developer cycles."

---

### [1:40 – 2:20] 4. ML Risk Prior & Honest Benchmark Results (40 seconds)
**Screen Action:**
1. Scroll to the **"Risk Assessment Prior"** card on the screen.
2. Highlight the ML model probability bar and the confidence bracket.
3. Switch briefly to the evaluation summary chart or show the benchmark comparison table in the documentation.

**Voiceover:**
> "To intelligently prioritize findings, PatchPilot uses a Gradient Boosting classifier trained on non-test Python functions from PyCode-Vul. But unlike black-box claims, we evaluated this rigorously.  
> When tested against thousands of held-out functions, static analysis alone reaches 90.5% recall, but with a 49% false-positive rate driven by style rules. When we apply the ML prior as a gate, false positives drop from 49% down to 9.3%, while maintaining balanced precision.  
> We also audited where Bandit excels: Bandit alone achieves 0.768 F1 on commit-mined datasets, but 86% of its true positives came from generic constructs rather than the specific commit flaw—the top rules among true positives were B110 (try/except/pass, 181), B101 (assert, 149), B108 (hardcoded /tmp, 73), B311 (random, 49), B602 (subprocess shell=True, 47), and B324 (md5/sha1, 37). PatchPilot transparently reports both, combining static breadth with ML precision."

---

### [2:20 – 2:45] 5. Limitations & What Comes Next (25 seconds)
**Screen Action:**
1. Scroll to the sidebar showing the **System Status & Privacy Notice**.
2. Return to the main overview screen.

**Voiceover:**
> "Code is analysed in temporary files that are deleted immediately and PatchPilot does not store it. If you enable LLM explanations, the code and findings are sent to Google's Gemini API, so do not use private code without permission. When no key is configured, static and ML analysis proceed locally.  
> While currently focused on single-file Python analysis, our roadmap includes multi-language cross-file taint tracking and automated pull request comments via GitHub Actions.  
> PatchPilot: actionable, explainable, and honest automated code review. Thank you!"
