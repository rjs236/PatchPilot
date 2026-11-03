"""GitHub Pull Request client for fetching metadata, changed files, and contents."""

import datetime
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from patchpilot.diff_utils import parse_patch_changed_lines

MAX_FILES_PER_PR = 20
MAX_CHANGED_LINES_PER_PR = 2000
DEFAULT_TIMEOUT_SECONDS = 10.0

PR_URL_PATTERN = re.compile(
    r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)(?:/(?:files|commits)?)?/?$"
)


class GitHubPRError(Exception):
    """Base exception for GitHub Pull Request operations."""


class GitHubInvalidURLError(GitHubPRError):
    """Raised when the PR URL is malformed or not a GitHub URL."""


class GitHubPRNotFoundError(GitHubPRError):
    """Raised when the PR or repository is not found (404) or is private."""


class GitHubRateLimitError(GitHubPRError):
    """Raised when GitHub API rate limits are exceeded (403/429)."""

    def __init__(
        self,
        message: str,
        remaining: int = 0,
        reset_epoch: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.remaining = remaining
        self.reset_epoch = reset_epoch


class GitHubNetworkError(GitHubPRError):
    """Raised when GitHub is unreachable due to timeout or network failures."""


@dataclass
class ChangedFile:
    """Represents a changed file within a pull request."""

    filename: str
    status: str  # added | modified | renamed | removed
    patch: Optional[str] = None
    changed_lines: Set[int] = field(default_factory=set)
    content: Optional[str] = None
    skip_reason: Optional[str] = None


@dataclass
class PullRequestMetadata:
    """Metadata for a fetched GitHub pull request."""

    owner: str
    repo: str
    pull_number: int
    title: str
    author: str
    base_branch: str
    head_branch: str
    head_sha: str
    html_url: str


def parse_pr_url(url: str) -> Tuple[str, str, int]:
    """Parse and validate a GitHub pull request URL.

    Accepts:
    - https://github.com/owner/repo/pull/123
    - https://github.com/owner/repo/pull/123/
    - https://github.com/owner/repo/pull/123/files
    - https://github.com/owner/repo/pull/123/commits
    - http:// variants

    Returns:
        (owner, repo, pull_number)

    Raises:
        GitHubInvalidURLError: If the URL is not a valid GitHub PR URL.
    """
    if not url or not isinstance(url, str):
        raise GitHubInvalidURLError("A GitHub Pull Request URL is required.")

    cleaned = url.strip()
    try:
        parsed = urlparse(cleaned)
    except Exception as exc:
        raise GitHubInvalidURLError(f"Invalid URL structure: {exc}") from exc

    hostname = (parsed.hostname or "").lower()
    if hostname != "github.com" and hostname != "www.github.com":
        raise GitHubInvalidURLError(
            f"Expected a github.com URL, but received hostname: '{hostname or 'empty'}'"
        )

    match = PR_URL_PATTERN.match(parsed.path)
    if not match:
        raise GitHubInvalidURLError(
            "Malformed GitHub PR URL. Expected format: 'https://github.com/owner/repo/pull/NUMBER'"
        )

    owner = match.group("owner")
    repo = match.group("repo")
    number = int(match.group("number"))
    return owner, repo, number


def _create_github_session(token: Optional[str] = None) -> requests.Session:
    """Create a configured requests Session with automatic 1-retry backoff and auth headers."""
    session = requests.Session()
    retries = Retry(
        total=1,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Resolve token without ever exposing it
    auth_token = token or os.environ.get("GITHUB_TOKEN")
    session.headers.update(
        {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PatchPilot-CodeReviewer/1.0",
        }
    )
    if auth_token and isinstance(auth_token, str) and auth_token.strip():
        session.headers["Authorization"] = f"Bearer {auth_token.strip()}"

    return session


def _handle_response_errors(response: requests.Response) -> None:
    """Examine HTTP response status code and raise appropriate clean domain exceptions.

    Guarantees no tokens are leaked in any exception message.
    """
    status = response.status_code
    if status < 400:
        return

    # 404: Not found or private repository
    if status == 404:
        raise GitHubPRNotFoundError(
            "Pull request or repository not found on GitHub. Verify the URL is correct "
            "and that the repository is public (or provide a GITHUB_TOKEN for private repos)."
        )

    # 403 or 429: Rate limit or forbidden
    if status in (403, 429):
        remaining_raw = response.headers.get("x-ratelimit-remaining", "0")
        reset_raw = response.headers.get("x-ratelimit-reset")
        try:
            remaining = int(remaining_raw)
        except (ValueError, TypeError):
            remaining = 0

        reset_time_str = "soon"
        reset_epoch = None
        if reset_raw:
            try:
                reset_epoch = int(reset_raw)
                dt = datetime.datetime.fromtimestamp(reset_epoch, tz=datetime.timezone.utc)
                reset_time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except (ValueError, TypeError, OverflowError):
                pass

        if remaining == 0 or status == 429:
            raise GitHubRateLimitError(
                f"GitHub API rate limit exceeded. Remaining quota: {remaining}. "
                f"Rate limit resets at: {reset_time_str}. Unauthenticated requests are limited "
                "to 60 requests per hour. Provide a GITHUB_TOKEN to increase your limit to 5,000/hr.",
                remaining=remaining,
                reset_epoch=reset_epoch,
            )
        raise GitHubPRError(f"GitHub API access forbidden (HTTP {status}).")

    if status >= 500:
        raise GitHubNetworkError(
            f"GitHub service encountered an error (HTTP {status}). Please try again later."
        )

    raise GitHubPRError(f"GitHub API request failed with HTTP {status}.")


class GitHubPRClient:
    """Client for retrieving pull request metadata, changed files, and source contents."""

    def __init__(self, token: Optional[str] = None) -> None:
        # Do not store raw token in public attributes; encapsulate in session
        self._session = _create_github_session(token)

    def fetch_pr(
        self,
        url: str,
    ) -> Tuple[PullRequestMetadata, List[ChangedFile], List[Dict[str, str]]]:
        """Fetch pull request metadata and changed Python files up to resource limits.

        Returns:
            (metadata, eligible_files, skipped_files_info)
        """
        owner, repo, number = parse_pr_url(url)
        metadata = self._fetch_metadata(owner, repo, number)
        eligible_files, skipped_files = self._fetch_changed_files(
            owner, repo, number, metadata.head_sha
        )
        return metadata, eligible_files, skipped_files

    def _fetch_metadata(self, owner: str, repo: str, number: int) -> PullRequestMetadata:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
        try:
            resp = self._session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
        except requests.Timeout as exc:
            raise GitHubNetworkError(
                "Timed out while connecting to GitHub API (10s limit exceeded)."
            ) from None
        except requests.RequestException as exc:
            raise GitHubNetworkError(f"Network error communicating with GitHub: {exc}") from None

        _handle_response_errors(resp)
        data = resp.json()

        return PullRequestMetadata(
            owner=owner,
            repo=repo,
            pull_number=number,
            title=data.get("title", ""),
            author=data.get("user", {}).get("login", "unknown"),
            base_branch=data.get("base", {}).get("ref", "main"),
            head_branch=data.get("head", {}).get("ref", "head"),
            head_sha=data.get("head", {}).get("sha", ""),
            html_url=data.get("html_url", f"https://github.com/{owner}/{repo}/pull/{number}"),
        )

    def _fetch_changed_files(
        self, owner: str, repo: str, number: int, head_sha: str
    ) -> Tuple[List[ChangedFile], List[Dict[str, str]]]:
        """Fetch list of changed files across paginated GitHub endpoint and retrieve .py contents."""
        all_raw_files: List[Dict[str, Any]] = []
        page = 1

        while True:
            files_url = (
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files"
                f"?per_page=100&page={page}"
            )
            try:
                resp = self._session.get(files_url, timeout=DEFAULT_TIMEOUT_SECONDS)
            except requests.Timeout:
                raise GitHubNetworkError(
                    "Timed out while fetching pull request file list."
                ) from None
            except requests.RequestException as exc:
                raise GitHubNetworkError(f"Network error fetching files: {exc}") from None

            _handle_response_errors(resp)
            page_data = resp.json()
            if not isinstance(page_data, list) or len(page_data) == 0:
                break

            all_raw_files.extend(page_data)
            if len(page_data) < 100 or page >= 50:
                break
            page += 1

        eligible_files: List[ChangedFile] = []
        skipped_files: List[Dict[str, str]] = []
        cumulative_changed_lines = 0

        for file_info in all_raw_files:
            filename = file_info.get("filename", "")
            status = file_info.get("status", "modified")
            patch = file_info.get("patch")

            # Check 1: Must be a Python file (.py)
            if not filename.endswith(".py"):
                # Non-Python files are silently ignored from code review
                continue

            # Check 2: Skip removed files
            if status == "removed":
                skipped_files.append({"filename": filename, "reason": "File was deleted in this pull request."})
                continue

            # Check 3: Check for empty or binary diff patch
            if not patch or not isinstance(patch, str):
                skipped_files.append(
                    {"filename": filename, "reason": "No diff text available (binary or oversized file)."}
                )
                continue

            changed_lines = parse_patch_changed_lines(patch)
            file_changed_count = len(changed_lines)

            # Check 4: Enforce max 20 changed Python files limit
            if len(eligible_files) >= MAX_FILES_PER_PR:
                skipped_files.append(
                    {
                        "filename": filename,
                        "reason": f"Review limit exceeded (maximum {MAX_FILES_PER_PR} Python files per PR).",
                    }
                )
                continue

            # Check 5: Enforce max 2,000 changed lines limit
            if cumulative_changed_lines + file_changed_count > MAX_CHANGED_LINES_PER_PR:
                skipped_files.append(
                    {
                        "filename": filename,
                        "reason": (
                            f"Review limit exceeded ({cumulative_changed_lines + file_changed_count} changed lines "
                            f"exceeds maximum {MAX_CHANGED_LINES_PER_PR} lines per PR)."
                        ),
                    }
                )
                continue

            # Fetch file contents at PR head SHA
            content, skip_err = self._fetch_file_content(owner, repo, filename, head_sha)
            if skip_err:
                skipped_files.append({"filename": filename, "reason": skip_err})
                continue

            eligible_files.append(
                ChangedFile(
                    filename=filename,
                    status=status,
                    patch=patch,
                    changed_lines=changed_lines,
                    content=content,
                )
            )
            cumulative_changed_lines += file_changed_count

        return eligible_files, skipped_files

    def _fetch_file_content(
        self, owner: str, repo: str, filename: str, head_sha: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Fetch raw contents of a file at head_sha, ensuring clean UTF-8 decoding."""
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{head_sha}/{filename}"
        try:
            resp = self._session.get(raw_url, timeout=DEFAULT_TIMEOUT_SECONDS)
        except requests.Timeout:
            return None, "Timed out fetching file content from GitHub raw content service."
        except requests.RequestException as exc:
            return None, f"Network error downloading file content: {exc}"

        if resp.status_code == 404:
            # Fallback to GitHub Contents API
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filename}?ref={head_sha}"
            try:
                resp = self._session.get(
                    api_url,
                    headers={"Accept": "application/vnd.github.v3.raw"},
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                return None, f"Failed retrieving file contents: {exc}"

        if resp.status_code != 200:
            return None, f"Failed retrieving file contents (HTTP {resp.status_code})."

        try:
            content = resp.content.decode("utf-8")
            return content, None
        except UnicodeDecodeError:
            return None, "File content is not valid UTF-8."
