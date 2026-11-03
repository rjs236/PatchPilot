# Neural Storm 2026 Hackathon Submission Checklist

**Track:** Future Tech (FT-05: AI-Powered Automated Code Review)  
**Project:** PatchPilot  
**Hard Submission Deadline:** November 4, 2026, at 8:00 PM IST  
**Recommended Internal Target:** November 4, 2026, at 2:00 PM IST (6-hour safety buffer)

---

### Phase 1: Repository & Code Completeness
- [ ] **Git Remote & Default Branch:**
  - Verify that `main` is set as the default repository branch on GitHub.
  - Run `git branch` locally to ensure you are on `main` and all recent commits are pushed:
    ```bash
    git push -u origin main
    ```
- [ ] **Repository Visibility:**
  - Confirm the GitHub repository is set to **Public** in GitHub repository Settings.
  - Open an **Incognito / Private browser window** and verify the entire repo, README, and code are visible without login.
- [ ] **Secret Scan Verification:**
  - Confirm `.env` was never committed and is absent from the repo (`git log --all --full-history -- ".env"` returns empty).
  - Verify no live API tokens (`AIza...`, `ghp_...`, `sk-...`) exist in the repo or Git history.
- [ ] **License & Attribution:**
  - `LICENSE` file (MIT) present at repo root.
  - Dataset attribution for PyCode-Vul verified and present in `README.md`.

---

### Phase 2: Documentation & README Verification
- [ ] **README Structure:**
  - [ ] Project title and track identifier (**FT-05: AI-Powered Automated Code Review**).
  - [ ] Team Name & Member Names updated (replace `[TEAM NAME]` and `[MEMBER NAMES]` placeholders).
  - [ ] Problem description, motivation, and solution approach clearly explained.
  - [ ] Three-tier architecture with Mermaid diagram.
  - [ ] Complete feature list: Snippet review, GitHub PR diff filtering, Offline sample data mode, Categorized findings.
  - [ ] Technology stack table (Python 3.11+, Streamlit, FastAPI, scikit-learn, Bandit, Ruff, Google Gemini 2.5).
  - [ ] Local installation and quick-start instructions tested step-by-step.
  - [ ] Frozen benchmark tables and honest evaluation included in exact required order:
    1. One-paragraph honest summary
    2. Primary benchmark comparison table
    3. Bandit audit limitation (86% generic construct findings)
    4. Excluded unparseable sensitivity table & analysis
    5. Exploratory security-and-bug-only table
    6. Test-function discovery and fix documentation
  - [ ] Responsible use & limitations clearly outlined (temporary files privacy, human-in-the-loop validation).
  - [ ] AI tools disclosure documented (Google Antigravity agent, Google Gemini 2.5).
- [ ] **Screenshots & Visuals:**
  - [ ] Capture and place `docs/screenshots/ui_overview.png` (Streamlit main interface).
  - [ ] Capture and place `docs/screenshots/snippet_analysis.png` (Snippet vulnerability finding with Gemini explanation).
  - [ ] Capture and place `docs/screenshots/github_pr_review.png` (PR changed-line findings vs hidden pre-existing).
  - [ ] (Optional / Recommended) Add `docs/screenshots/demo_walkthrough.gif`.

---

### Phase 3: Deployment & Live Demo (Streamlit / Hugging Face Spaces)
- [ ] **Deploy Streamlit Application:**
  - Deploy to **Hugging Face Spaces** (Streamlit SDK) or **Streamlit Community Cloud** (see `docs/DEPLOYMENT.md`).
- [ ] **Environment Secrets on Platform:**
  - Add `GEMINI_API_KEY` (and optional `GITHUB_TOKEN`) in the deployment platform's secret settings—**never in code**.
- [ ] **Incognito Testing:**
  - Open the live deployment URL in an **Incognito / Private browser window**:
    - Verify page loads with the sidebar showing healthy system components.
    - Test Snippet Analysis without an API key (confirm offline fallback works).
    - Test Snippet Analysis with Gemini enabled (confirm explanation generates).
    - Test "Load Offline Demo PR (Sample Data)" (confirm diff filtering works).
- [ ] **Update README Link:**
  - Update `[Live Demo URL]` placeholder in `README.md` with the live deployment link.

---

### Phase 4: Video Demo & Presentation
- [ ] **Record Demo Video:**
  - Follow `docs/DEMO_SCRIPT.md` strictly (keep total time between 2:30 and 3:00 minutes).
  - Clearly cover:
    1. Problem & FT-05 track overview (20s)
    2. Snippet review with Bandit/AST + Gemini explanation & fix (40s)
    3. PR review with diff-aware changed-line isolation (40s)
    4. ML risk prior & honest evaluation numbers / Bandit audit (40s)
    5. Limitations, privacy posture, and next steps (20s)
- [ ] **Upload Video:**
  - Upload to YouTube (Unlisted or Public) or Google Drive (set to *"Anyone with the link can view"*).
  - Test video link in an **Incognito browser window** to ensure audio, video, and permissions work smoothly.

---

### Phase 5: Submission Form Preparation
- [ ] **Prepare Submission Fields in Advance:**
  - **Project Name:** PatchPilot
  - **Track:** Future Tech (FT-05: AI-Powered Automated Code Review)
  - **Public GitHub Repository Link:** `https://github.com/<your-username>/<repo-name>`
  - **Live Demo Link:** `https://huggingface.co/spaces/<username>/patchpilot` (or Streamlit Cloud)
  - **Demo Video Link:** YouTube / Drive URL
  - **Short Elevator Pitch / Description:** (Pre-draft 200 words from README)
  - **Team Members:** Full names, emails, and roles.
- [ ] **Final Submission Execution:**
  - Submit before 2:00 PM IST on November 4, 2026, to avoid server congestion and deadline stress.
  - Save a screenshot and confirmation email of the successful submission.
