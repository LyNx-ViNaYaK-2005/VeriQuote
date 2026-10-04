"""Export only public conversation content, never session state or credentials."""
from datetime import datetime, timezone
import re

from src.models import Turn


def escape(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", text)


def export_chat(history: list[Turn], documents: list[str], *, format: str = "md",
                timestamp: datetime | None = None) -> str:
    if format not in ("md", "txt"):
        raise ValueError("Export format must be md or txt")
    stamp = (timestamp or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    names = list(dict.fromkeys([*documents, *[c.source for t in history for c in t.answer.citations]]))
    lines = ["# Folio · Chat export", "", f"Exported: {stamp}", "", "## Documents", ""]
    lines += [f"- {escape(name)}" for name in names] or ["- None"]
    lines += ["", "Historical citations may refer to documents removed from the active workspace.", ""]
    for turn in history:
        lines += ["---", "", "### User", "", escape(turn.question), "", "### Assistant", "", turn.answer.text, ""]
        if turn.answer.citations:
            lines += ["Sources:", ""]
            lines += [f"- [{c.number}] {escape(c.source)} — Page {c.page}" for c in turn.answer.citations]
            lines.append("")
    result = "\n".join(lines)
    if format == "txt":
        result = re.sub(r"(?m)^#{1,3} ", "", result)
        result = re.sub(r"\\([\\`*_{}\[\]<>#!|])", r"\1", result)
    return result
