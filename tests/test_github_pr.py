"""Unit tests for GitHub Pull Request parsing, client, security, and API endpoints."""

import io
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from patchpilot.api import app
from patchpilot.github_pr import (
    GitHubInvalidURLError,
    GitHubNetworkError,
    GitHubPRClient,
    GitHubPRNotFoundError,
    GitHubRateLimitError,
    parse_pr_url,
)
from patchpilot.pipeline import load_demo_pr, review_pull_request


# ==============================================================================
# 1. URL Parsing Tests
# ==============================================================================


def test_parse_pr_url_valid_standard():
    owner, repo, number = parse_pr_url("https://github.com/psf/requests/pull/1234")
    assert owner == "psf"
    assert repo == "requests"
    assert number == 1234


def test_parse_pr_url_valid_trailing_slash():
    owner, repo, number = parse_pr_url("https://github.com/pallets/flask/pull/5678/")
    assert owner == "pallets"
    assert repo == "flask"
    assert number == 5678


def test_parse_pr_url_valid_files_and_commits_suffixes():
    o1, r1, n1 = parse_pr_url("https://github.com/django/django/pull/999/files")
    assert (o1, r1, n1) == ("django", "django", 999)

    o2, r2, n2 = parse_pr_url("https://github.com/django/django/pull/999/commits")
    assert (o2, r2, n2) == ("django", "django", 999)


def test_parse_pr_url_rejects_non_github():
    with pytest.raises(GitHubInvalidURLError, match="Expected a github.com URL"):
        parse_pr_url("https://gitlab.com/owner/repo/pull/123")

    with pytest.raises(GitHubInvalidURLError, match="Expected a github.com URL"):
        parse_pr_url("https://bitbucket.org/owner/repo/pull/123")


def test_parse_pr_url_rejects_malformed_and_missing_number():
    with pytest.raises(GitHubInvalidURLError):
        parse_pr_url("https://github.com/owner/repo")

    with pytest.raises(GitHubInvalidURLError):
        parse_pr_url("https://github.com/owner/repo/pull/")

    with pytest.raises(GitHubInvalidURLError):
        parse_pr_url("https://github.com/owner/repo/pull/abc")

    with pytest.raises(GitHubInvalidURLError):
        parse_pr_url("")


# ==============================================================================
# 2. Token Security Assertion: Token NEVER appears in errors or logs
# ==============================================================================


def test_token_never_leaked_in_exceptions_or_string_representations(caplog):
    secret_token = "ghp_VERY_SECRET_TOKEN_XYZ_123456789"
    client = GitHubPRClient(token=secret_token)

    # 1. Simulate 404 error
    mock_resp_404 = MagicMock(spec=requests.Response)
    mock_resp_404.status_code = 404
    mock_resp_404.headers = {}

    with patch.object(client._session, "get", return_value=mock_resp_404):
        with pytest.raises(GitHubPRNotFoundError) as exc_info:
            client.fetch_pr("https://github.com/owner/repo/pull/10")
        assert secret_token not in str(exc_info.value)
        assert secret_token not in repr(exc_info.value)

    # 2. Simulate 403 Rate Limit error
    mock_resp_403 = MagicMock(spec=requests.Response)
    mock_resp_403.status_code = 403
    mock_resp_403.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1893456000"}

    with patch.object(client._session, "get", return_value=mock_resp_403):
        with pytest.raises(GitHubRateLimitError) as exc_info:
            client.fetch_pr("https://github.com/owner/repo/pull/10")
        assert secret_token not in str(exc_info.value)
        assert secret_token not in repr(exc_info.value)

    # 3. Simulate Network Error
    with patch.object(client._session, "get", side_effect=requests.RequestException("connection failed")):
        with pytest.raises(GitHubNetworkError) as exc_info:
            client.fetch_pr("https://github.com/owner/repo/pull/10")
        assert secret_token not in str(exc_info.value)

    # Assert captured logs never contain the secret token
    assert secret_token not in caplog.text


# ==============================================================================
# 3. GitHub PR Client Behavior Tests (Mocked Offline)
# ==============================================================================


def test_client_handles_rate_limit_and_quota_reset_headers():
    client = GitHubPRClient()
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 403
    mock_resp.headers = {
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": "1735689600",  # 2025-01-01 00:00:00 UTC
    }

    with patch.object(client._session, "get", return_value=mock_resp):
        with pytest.raises(GitHubRateLimitError) as exc_info:
            client.fetch_pr("https://github.com/owner/repo/pull/1")

        err_msg = str(exc_info.value)
        assert "rate limit exceeded" in err_msg.lower()
        assert "remaining quota: 0" in err_msg.lower()
        assert "60 requests per hour" in err_msg


def test_client_handles_timeout_and_network_error():
    client = GitHubPRClient()
    with patch.object(client._session, "get", side_effect=requests.Timeout("Connection timed out")):
        with pytest.raises(GitHubNetworkError, match="Timed out while connecting to GitHub API"):
            client.fetch_pr("https://github.com/owner/repo/pull/1")


def test_client_pagination_and_file_filtering():
    client = GitHubPRClient()

    # Mock PR Metadata
    meta_resp = MagicMock(spec=requests.Response)
    meta_resp.status_code = 200
    meta_resp.json.return_value = {
        "title": "feat: add analytics",
        "user": {"login": "alice"},
        "base": {"ref": "main"},
        "head": {"ref": "feat-branch", "sha": "abcd1234efgh"},
        "html_url": "https://github.com/owner/repo/pull/1",
    }

    # Page 1 files (100 files to trigger page 2 check)
    page1_items = [{"filename": f"module_{i}.py", "status": "modified", "patch": "@@ -1 +1 @@\n+x=1"} for i in range(100)]
    files_page1 = MagicMock(spec=requests.Response)
    files_page1.status_code = 200
    files_page1.json.return_value = page1_items

    # Page 2 files (empty to stop pagination)
    files_page2 = MagicMock(spec=requests.Response)
    files_page2.status_code = 200
    files_page2.json.return_value = []

    # File content response
    content_resp = MagicMock(spec=requests.Response)
    content_resp.status_code = 200
    content_resp.content = b"x = 1\n"

    def mock_get(url, *args, **kwargs):
        if url.endswith("/pulls/1"):
            return meta_resp
        if "&page=1" in url:
            return files_page1
        if "&page=2" in url:
            return files_page2
        if "raw.githubusercontent.com" in url or "contents" in url:
            return content_resp
        return MagicMock(status_code=404)

    with patch.object(client._session, "get", side_effect=mock_get):
        metadata, eligible, skipped = client.fetch_pr("https://github.com/owner/repo/pull/1")

    assert metadata.title == "feat: add analytics"
    assert metadata.author == "alice"
    # Max files cap is 20
    assert len(eligible) == 20
    # Remaining 80 files should be recorded in skipped
    assert len(skipped) == 80
    assert "maximum 20 Python files" in skipped[0]["reason"]


def test_client_skips_removed_binary_and_non_utf8_files():
    client = GitHubPRClient()

    meta_resp = MagicMock(spec=requests.Response)
    meta_resp.status_code = 200
    meta_resp.json.return_value = {
        "title": "test pr",
        "user": {"login": "bob"},
        "base": {"ref": "main"},
        "head": {"ref": "head", "sha": "1234"},
        "html_url": "https://github.com/owner/repo/pull/2",
    }

    files_resp = MagicMock(spec=requests.Response)
    files_resp.status_code = 200
    files_resp.json.return_value = [
        {"filename": "deleted.py", "status": "removed", "patch": None},
        {"filename": "binary.py", "status": "modified", "patch": None},
        {"filename": "invalid_utf8.py", "status": "modified", "patch": "@@ -1 +1 @@\n+test"},
        {"filename": "valid.py", "status": "modified", "patch": "@@ -1 +1 @@\n+a = 1"},
    ]

    def mock_get(url, *args, **kwargs):
        if url.endswith("/pulls/2"):
            return meta_resp
        if "pulls/2/files" in url:
            return files_resp
        if "invalid_utf8.py" in url:
            resp = MagicMock(spec=requests.Response)
            resp.status_code = 200
            resp.content = b"\x80abc invalid"
            return resp
        if "valid.py" in url:
            resp = MagicMock(spec=requests.Response)
            resp.status_code = 200
            resp.content = b"a = 1\n"
            return resp
        return MagicMock(status_code=404)

    with patch.object(client._session, "get", side_effect=mock_get):
        _, eligible, skipped = client.fetch_pr("https://github.com/owner/repo/pull/2")

    assert len(eligible) == 1
    assert eligible[0].filename == "valid.py"
    skipped_filenames = [s["filename"] for s in skipped]
    assert "deleted.py" in skipped_filenames
    assert "binary.py" in skipped_filenames
    assert "invalid_utf8.py" in skipped_filenames


def test_client_enforces_2000_lines_limit():
    client = GitHubPRClient()

    meta_resp = MagicMock(spec=requests.Response)
    meta_resp.status_code = 200
    meta_resp.json.return_value = {
        "title": "huge diff pr",
        "user": {"login": "carol"},
        "base": {"ref": "main"},
        "head": {"ref": "head", "sha": "1234"},
        "html_url": "https://github.com/owner/repo/pull/3",
    }

    # Generate a patch with 1,500 changed lines
    big_patch_1 = "@@ -0,0 +1,1500 @@\n" + "\n".join("+x=1" for _ in range(1500))
    # Generate another patch with 800 changed lines (1500 + 800 = 2300 > 2000)
    big_patch_2 = "@@ -0,0 +1,800 @@\n" + "\n".join("+y=1" for _ in range(800))

    files_resp = MagicMock(spec=requests.Response)
    files_resp.status_code = 200
    files_resp.json.return_value = [
        {"filename": "file1.py", "status": "added", "patch": big_patch_1},
        {"filename": "file2.py", "status": "added", "patch": big_patch_2},
    ]

    def mock_get(url, *args, **kwargs):
        if url.endswith("/pulls/3"):
            return meta_resp
        if "pulls/3/files" in url:
            return files_resp
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.content = b"x = 1\n"
        return resp

    with patch.object(client._session, "get", side_effect=mock_get):
        _, eligible, skipped = client.fetch_pr("https://github.com/owner/repo/pull/3")

    assert len(eligible) == 1
    assert eligible[0].filename == "file1.py"
    assert len(skipped) == 1
    assert skipped[0]["filename"] == "file2.py"
    assert "exceeds maximum 2000 lines" in skipped[0]["reason"]


# ==============================================================================
# 4. Pipeline & Demo Mode Tests
# ==============================================================================


def test_load_demo_pr_offline():
    demo = load_demo_pr()
    assert "pr_metadata" in demo
    assert demo["pr_metadata"]["owner"] == "octocat"
    assert demo["pr_metadata"]["repo"] == "demo-repo"
    assert len(demo["files"]) >= 2
    assert demo["summary"]["total"] >= 4
    assert demo["pre_existing_hidden_count"] > 0

    # Verify findings contain pre-generated sample label
    all_explanations = [
        f.get("explanation", "")
        for file_obj in demo["files"]
        for f in file_obj["findings"]
    ]
    assert any("[pre-generated sample]" in exp for exp in all_explanations)


def test_pipeline_review_pull_request_filters_unchanged_lines():
    mock_meta = MagicMock(
        owner="owner",
        repo="repo",
        pull_number=10,
        title="Sample PR",
        author="alice",
        base_branch="main",
        head_branch="patch-1",
        head_sha="sha123",
        html_url="https://github.com/owner/repo/pull/10",
    )
    # File has 20 lines.
    # Line 2 has mutable default arg: def foo(x=[]):
    # Line 15 has eval(cmd)
    code = (
        "import os\n"
        "def foo(x=[]):\n"  # line 2 (unchanged context)
        "    pass\n"
        "def bar():\n"
        "    return 1\n"
        "\n\n\n\n\n\n\n\n\n"
        "def run_query(cmd):\n"
        "    return eval(cmd)\n"  # line 15 (changed in PR!)
    )
    # Diff only touches line 16 (return eval(cmd))
    sample_diff = "@@ -15,2 +15,2 @@\n def run_query(cmd):\n+    return eval(cmd)\n"
    from patchpilot.github_pr import ChangedFile

    changed_file = ChangedFile(
        filename="app.py",
        status="modified",
        patch=sample_diff,
        changed_lines={16},
        content=code,
    )

    with patch.object(GitHubPRClient, "fetch_pr", return_value=(mock_meta, [changed_file], [])):
        result = review_pull_request(
            url="https://github.com/owner/repo/pull/10",
            use_llm=False,
            use_ml=False,
        )

    assert result["pr_metadata"]["title"] == "Sample PR"
    assert len(result["files"]) == 1
    file_findings = result["files"][0]["findings"]
    # Only eval on line 16 should be reported; mutable default on line 2 should be filtered out!
    assert len(file_findings) == 1
    assert file_findings[0]["line"] == 16
    assert result["pre_existing_hidden_count"] >= 1
    assert file_findings[0]["rule_id"] == "eval-exec"
    assert result["pre_existing_hidden_count"] >= 1


# ==============================================================================
# 5. FastAPI PR Endpoint Tests
# ==============================================================================


test_client = TestClient(app)


def test_api_demo_pr_endpoint():
    response = test_client.get("/review/pr/demo")
    assert response.status_code == 200
    data = response.json()
    assert data["pr_metadata"]["owner"] == "octocat"
    assert "files" in data
    assert data["summary"]["total"] >= 4


def test_api_review_pr_invalid_url():
    response = test_client.post("/review/pr", json={"url": "not-a-url"})
    assert response.status_code == 400
    assert "Expected a github.com URL" in response.json()["detail"]


def test_api_review_pr_not_found():
    with patch("patchpilot.api.review_pull_request", side_effect=GitHubPRNotFoundError("PR not found")):
        response = test_client.post("/review/pr", json={"url": "https://github.com/owner/nonexistent/pull/1"})
    assert response.status_code == 404
    assert "PR not found" in response.json()["detail"]


def test_api_review_pr_rate_limited():
    with patch("patchpilot.api.review_pull_request", side_effect=GitHubRateLimitError("Rate limit exceeded", remaining=0)):
        response = test_client.post("/review/pr", json={"url": "https://github.com/owner/repo/pull/1"})
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]


def test_api_review_pr_network_error():
    with patch("patchpilot.api.review_pull_request", side_effect=GitHubNetworkError("Timeout")):
        response = test_client.post("/review/pr", json={"url": "https://github.com/owner/repo/pull/1"})
    assert response.status_code == 502
    assert "Timeout" in response.json()["detail"]


def test_api_review_pr_success():
    demo = load_demo_pr()
    with patch("patchpilot.api.review_pull_request", return_value=demo):
        response = test_client.post(
            "/review/pr",
            json={"url": "https://github.com/octocat/demo-repo/pull/42", "use_llm": False, "use_ml": True},
        )
    assert response.status_code == 200
    assert response.json()["pr_metadata"]["pull_number"] == 42
