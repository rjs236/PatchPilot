# PatchPilot Deployment Guide

This guide provides step-by-step instructions for deploying PatchPilot to **Hugging Face Spaces** (recommended primary target) and **Streamlit Community Cloud** (alternative).

---

## 1. Pre-Deployment Readiness & Artifact Verification

- **Entry Point:** `app/streamlit_app.py`
- **Model Artifacts:**
  - `ml/baseline_model.joblib`: 798 KB
  - `ml/baseline_model_clean.joblib`: 798 KB
  - Both model files are committed directly to Git. **No dataset downloads occur at startup or runtime.**
- **Runtime Tools:**
  - `bandit` and `ruff` are listed in `requirements.txt` and installed into the environment at build time.
- **Secrets Management:**
  - `GEMINI_API_KEY` and `GITHUB_TOKEN` are read strictly from platform secrets or environment variables.
  - Zero-key fallback: With zero secrets configured, AST checks, Bandit, Ruff, the ML risk model, and offline demo PR review remain 100% operational.
- **Safety Limits & Privacy:**
  - 200,000 character maximum input cap.
  - Subprocess timeouts (5 seconds max for Bandit and Ruff).
  - Session quota limit: Max 20 Gemini calls per user session.
  - Zero persistence: Submitted code is analyzed in-memory and discarded immediately. No code is stored, cached, or logged.

---

## 2. Option A: Hugging Face Spaces (Streamlit SDK)

Hugging Face Spaces provides a free, self-contained container with public URL access.

### The README Front-Matter Trade-off
Hugging Face Spaces uses YAML front matter at the very top of `README.md` to configure the Space:
```yaml
---
title: PatchPilot
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.42.0
app_file: app/streamlit_app.py
pinned: false
---
```

**Trade-off Comparison:**
1. **Direct GitHub Sync (Single Repo):**
   - *Pros:* One single repository to push to. Hugging Face automatically updates on every commit.
   - *Cons:* GitHub renders the YAML header block at the top of the README. While standard on Hugging Face, some hackathon evaluators prefer a pristine GitHub landing page.
2. **Dedicated Hugging Face Space Git Remote (Recommended for Hackathons):**
   - *Pros:* Keeps the GitHub repository README completely clean for hackathon judges, while Hugging Face gets its custom front-matter.
   - *Cons:* Requires pushing to two Git remotes (`origin` for GitHub and `hf` for Hugging Face).

---

### Click-by-Click Deployment on Hugging Face Spaces

#### Step 1: Create Space
1. Log in to [Hugging Face](https://huggingface.co).
2. Click on your profile icon in the top right and select **"New Space"**.
3. Fill out the Space creation form:
   - **Space Name:** `patchpilot`
   - **License:** `mit`
   - **Select SDK:** Choose **Streamlit**.
   - **Hardware:** Select **CPU basic (free)** (PatchPilot's ML inference takes ~15ms and runs efficiently on basic CPU).
   - **Space visibility:** Select **Public**.
4. Click **"Create Space"**.

#### Step 2: Configure Environment Secrets
1. In your newly created Space, navigate to the **Settings** tab.
2. Scroll to the **"Variables and secrets"** section.
3. Under **"New secret"**, add:
   - Name: `GEMINI_API_KEY`
   - Value: `<Your Google Gemini API Key>`
4. (Optional) Add `GITHUB_TOKEN` for higher GitHub API rate limits (5,000 requests/hr instead of 60/hr).

#### Step 3: Push Code to Hugging Face
In your local terminal, add the Hugging Face Space as a second git remote and push:
```bash
# Add Hugging Face git remote (replace <username> with your HF username)
git remote add hf https://huggingface.co/spaces/<username>/patchpilot

# If using a separate HF README with front matter, push your branch:
git push hf main
```
*Note: If your Space requires the YAML front matter in README.md, Hugging Face provides a web editor to prepend the YAML block directly in their web UI under "Files".*

---

## 3. Option B: Streamlit Community Cloud

Streamlit Community Cloud natively connects to public GitHub repositories and requires no README front matter.

### Click-by-Click Steps

1. Log in to [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
2. Click **"New app"**.
3. Configure repository settings:
   - **Repository:** `<your-github-username>/PatchPilot`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** Customize if desired (e.g., `patchpilot-review.streamlit.app`).
4. Click **"Advanced settings..."** before deploying:
   - Select **Python 3.11** or **Python 3.12**.
   - In the **Secrets** TOML editor, enter:
     ```toml
     GEMINI_API_KEY = "your_gemini_api_key_here"
     GITHUB_TOKEN = "your_optional_github_token_here"
     ```
5. Click **"Save"**, then click **"Deploy!"**.
6. The build log will run `pip install -r requirements.txt` (installing Streamlit, Bandit, Ruff, scikit-learn, and Google GenAI) and launch the app in 1–2 minutes.

---

## 4. Verification After Deployment

Once deployed, open the live URL in an **Incognito / Private browser window**:
1. **Sidebar Verification:** Confirm the **System Status & Active Layers** indicators show green checkmarks for AST, Bandit, Ruff, and ML Risk Model.
2. **Offline Snippet Test:** Paste a snippet with `shell=True` and click "Analyze Code Snippet". Verify Bandit B602 and AST findings display immediately.
3. **Gemini Test:** If an API key was configured, verify the AI explanation and remediation card render properly.
4. **Demo PR Test:** Click the "GitHub Pull Request" tab and click "Load Offline Demo PR (Sample Data)". Confirm the diff filtering correctly highlights line 42 and tucks line 12 into pre-existing issues.
