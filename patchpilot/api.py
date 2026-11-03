"""FastAPI service exposing PatchPilot code review and GitHub PR review capabilities."""

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from patchpilot.github_pr import (
    GitHubInvalidURLError,
    GitHubNetworkError,
    GitHubPRNotFoundError,
    GitHubRateLimitError,
    GitHubPRError,
)
from patchpilot.pipeline import load_demo_pr, review, review_pull_request

app = FastAPI(
    title="PatchPilot Code Review API",
    description="Automated AI-powered code review combining static analysis, ML risk priors, and Gemini LLM.",
    version="0.2.0",
)

# Enable CORS for local dashboards and integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    """Payload for submitting code to be reviewed."""

    code: str = Field(..., max_length=200000, description="Python source code (max 200,000 characters).")
    filename: str = Field("snippet.py", description="Source file name.")
    use_llm: bool = Field(False, description="Whether to invoke Gemini LLM enrichment.")
    use_ml: bool = Field(True, description="Whether to invoke ML function risk classifier.")


class PullRequestReviewRequest(BaseModel):
    """Payload for submitting a GitHub pull request URL for review."""

    url: str = Field(..., description="Public GitHub Pull Request URL.")
    use_llm: bool = Field(False, description="Whether to invoke Gemini LLM enrichment.")
    use_ml: bool = Field(True, description="Whether to invoke ML function risk classifier.")


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/review")
def run_review(req: ReviewRequest) -> Dict[str, Any]:
    """Execute code review across static analysis, ranking, and optional LLM layers."""
    if len(req.code) > 200000:
        raise HTTPException(
            status_code=400,
            detail="Code exceeds the maximum allowed size of 200,000 characters.",
        )
    return review(code=req.code, filename=req.filename, use_llm=req.use_llm, use_ml=req.use_ml)


@app.post("/review/pr")
def run_pr_review(req: PullRequestReviewRequest) -> Dict[str, Any]:
    """Execute code review on changed Python lines of a GitHub pull request."""
    try:
        return review_pull_request(
            url=req.url,
            use_llm=req.use_llm,
            use_ml=req.use_ml,
        )
    except GitHubInvalidURLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GitHubPRNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except (GitHubNetworkError, GitHubPRError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/review/pr/demo")
def get_demo_pr_review() -> Dict[str, Any]:
    """Return bundled offline sample PR review without calling external APIs."""
    return load_demo_pr()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("patchpilot.api:app", host="0.0.0.0", port=8000, reload=True)
