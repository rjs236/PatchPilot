"""Data models for PatchPilot analyzers and review pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SeverityType = Literal["low", "medium", "high"]
CategoryType = Literal["security", "bug", "complexity", "style"]


@dataclass
class Finding:
    """Represents a code finding detected by static analysis, ML, or LLM."""

    tool: str
    rule_id: str
    category: str  # security | bug | complexity | style
    severity: str  # low | medium | high
    message: str
    line: int
    end_line: int
    confidence: float  # 0.0 to 1.0
    score: float = 0.0
    explanation: str = ""
    suggested_fix: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert finding into a JSON-serializable dictionary."""
        return asdict(self)
