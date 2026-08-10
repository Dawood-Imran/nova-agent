from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from .tools import WorkspaceTools

REFERENCE_PATTERN = re.compile(r'(?<!\S)@(?:"([^"\n]+)"|([^\s]+))')
SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}


def extract_file_references(prompt: str) -> list[str]:
    """Extract unique @path and @"path with spaces" references in input order."""
    references: list[str] = []
    for match in REFERENCE_PATTERN.finditer(prompt):
        path = match.group(1) or match.group(2)
        if path not in references:
            references.append(path)
    return references


def _is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return name in SENSITIVE_NAMES or name.startswith(".env.")


def _resolve_reference(workspace: Path, reference: str) -> Path | None:
    candidate = (workspace / reference).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    return candidate


def build_referenced_file_context(
    workspace: str | Path,
    prompt: str,
    *,
    max_file_bytes: int = 20_000,
    max_total_bytes: int = 60_000,
) -> str:
    """Build bounded, structured context for files explicitly tagged with @."""
    root = Path(workspace).expanduser().resolve()
    sections: list[str] = []
    included_bytes = 0

    for reference in extract_file_references(prompt):
        escaped_reference = html.escape(reference, quote=True)
        resolved = _resolve_reference(root, reference)
        if resolved is None:
            sections.append(
                f'<referenced_file path="{escaped_reference}" content_included="false" '
                'reason="outside_workspace" />'
            )
            continue
        if _is_sensitive(resolved):
            sections.append(
                f'<referenced_file path="{escaped_reference}" content_included="false" '
                'reason="sensitive_file" />'
            )
            continue
        if not resolved.is_file():
            sections.append(
                f'<referenced_file path="{escaped_reference}" content_included="false" '
                'reason="not_found_or_not_file" />'
            )
            continue

        size = resolved.stat().st_size
        if size > max_file_bytes or included_bytes + size > max_total_bytes:
            sections.append(
                f'<referenced_file path="{escaped_reference}" content_included="false" '
                f'reason="file_too_large" size_bytes="{size}" />'
            )
            continue
        try:
            content = resolved.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            sections.append(
                f'<referenced_file path="{escaped_reference}" content_included="false" '
                f'reason="binary_or_unreadable" size_bytes="{size}" />'
            )
            continue

        included_bytes += size
        sections.append(
            f'<referenced_file path="{escaped_reference}" content_included="true">\n'
            f"{content}\n"
            "</referenced_file>"
        )

    if not sections:
        return ""
    return "Explicitly referenced workspace files:\n" + "\n".join(sections)


class WorkspaceFileCompleter(Completer):
    """Suggest safe workspace-relative file references after an @ character."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        tools = WorkspaceTools(self.workspace)
        self.paths = [
            path
            for path in tools.find_files("**/*", max_results=10_000)
            if not _is_sensitive(self.workspace / path)
        ]

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        del complete_event
        before_cursor = document.text_before_cursor
        match = re.search(r'(?:^|\s)@(?P<quoted>")?(?P<fragment>[^\s"]*)$', before_cursor)
        if not match:
            return

        fragment = match.group("fragment")
        quoted = bool(match.group("quoted"))
        for path in self.paths:
            if fragment.lower() not in path.lower():
                continue
            if quoted:
                replacement = f'{path}"'
            elif " " in path:
                replacement = f'"{path}"'
            else:
                replacement = path
            yield Completion(
                replacement,
                start_position=-len(fragment),
                display=path,
            )
