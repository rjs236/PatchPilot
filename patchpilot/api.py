"""FastAPI service exposing PatchPilot code review capabilities."""

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from patchpilot.pipeline import review

app = FastAPI(
    title="PatchPilot Code Review API",
    description="Automated AI-powered code review combining static analysis and Gemini LLM.",
    version="0.1.0",
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
    return review(code=req.code, filename=req.filename, use_llm=req.use_llm)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("patchpilot.api:app", host="0.0.0.0", port=8000, reload=True)
