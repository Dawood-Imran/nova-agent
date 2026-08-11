from __future__ import annotations

import difflib
import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReplacementInput(BaseModel):
    """One exact replacement in a Pi-compatible edit request."""

    model_config = ConfigDict(extra="forbid")

    oldText: str = Field(description="Exact text for one targeted replacement. It must be unique.")
    newText: str = Field(description="Replacement text for this targeted edit.")


class EditToolInput(BaseModel):
    """Validated and backward-compatible model-facing edit arguments."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Workspace-relative or absolute path to the file to edit.")
    edits: list[ReplacementInput] = Field(
        min_length=1,
        description="Non-overlapping replacements matched against the original file.",
    )

    @model_validator(mode="before")
    @classmethod
    def prepare_arguments(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value

        arguments = dict(value)
        if isinstance(arguments.get("edits"), str):
            try:
                parsed = json.loads(arguments["edits"])
            except (TypeError, ValueError):
                pass
            else:
                if isinstance(parsed, list):
                    arguments["edits"] = parsed

        old_text = arguments.get("oldText")
        new_text = arguments.get("newText")
        if isinstance(old_text, str) and isinstance(new_text, str):
            arguments.pop("oldText")
            arguments.pop("newText")
            edits = arguments.get("edits")
            if not isinstance(edits, list):
                edits = []
            arguments["edits"] = [*edits, {"oldText": old_text, "newText": new_text}]
        return arguments


@dataclass(frozen=True)
class MatchedEdit:
    index: int
    start: int
    length: int
    new_text: str


@dataclass(frozen=True)
class EditResult:
    replacements: int
    first_changed_line: int
    diff: str
    patch: str


_MUTATION_LOCKS: dict[Path, threading.Lock] = {}
_MUTATION_LOCKS_GUARD = threading.Lock()


@contextmanager
def file_mutation_lock(path: Path) -> Iterator[None]:
    """Serialize mutations to the same real file while allowing different files in parallel."""
    key = path.resolve()
    with _MUTATION_LOCKS_GUARD:
        lock = _MUTATION_LOCKS.setdefault(key, threading.Lock())
    with lock:
        yield


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def detect_line_ending(text: str) -> str:
    match = re.search(r"\r\n|\r|\n", text)
    return "\r\n" if match and match.group(0) == "\r\n" else "\n"


def restore_line_endings(text: str, line_ending: str) -> str:
    return text.replace("\n", "\r\n") if line_ending == "\r\n" else text


def count_occurrences(content: str, target: str) -> int:
    if not target:
        return 0
    count = 0
    offset = 0
    while True:
        index = content.find(target, offset)
        if index < 0:
            return count
        count += 1
        offset = index + len(target)


_CHARACTER_EQUIVALENTS = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "\u00a0": " ",
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
    }
)


def _canonicalize_with_positions(text: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        normalized = character.translate(_CHARACTER_EQUIVALENTS)
        if normalized == "\n":
            while characters and characters[-1] in {" ", "\t"}:
                characters.pop()
                positions.pop()
        characters.append(normalized)
        positions.append(index)
    while characters and characters[-1] in {" ", "\t"}:
        characters.pop()
        positions.pop()
    return "".join(characters), positions


def _find_unique_fuzzy_match(content: str, target: str) -> tuple[int, int] | None:
    canonical_content, positions = _canonicalize_with_positions(content)
    canonical_target, _ = _canonicalize_with_positions(target)
    if not canonical_target:
        return None

    matches: list[int] = []
    offset = 0
    while True:
        index = canonical_content.find(canonical_target, offset)
        if index < 0:
            break
        matches.append(index)
        offset = index + len(canonical_target)
    if len(matches) != 1:
        return None

    canonical_start = matches[0]
    canonical_end = canonical_start + len(canonical_target)
    original_start = positions[canonical_start]
    original_end = positions[canonical_end] if canonical_end < len(positions) else len(content)
    return original_start, original_end


def _coerce_replacement(edit: ReplacementInput | Mapping[str, str], index: int) -> tuple[str, str]:
    if isinstance(edit, ReplacementInput):
        return edit.oldText, edit.newText
    try:
        return edit["oldText"], edit["newText"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"edits[{index}] must contain string oldText and newText fields") from error


def apply_edits_to_content(
    original_content: str,
    edits: Sequence[ReplacementInput | Mapping[str, str]],
    display_path: str,
) -> str:
    """Apply non-overlapping replacements located against one original LF-normalized string."""
    if not edits:
        raise ValueError("edits must contain at least one replacement")

    matched: list[MatchedEdit] = []
    for edit_index, edit in enumerate(edits):
        old_text, new_text = _coerce_replacement(edit, edit_index)
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ValueError(f"edits[{edit_index}] oldText and newText must be strings")
        old_text = normalize_to_lf(old_text)
        new_text = normalize_to_lf(new_text)
        if not old_text:
            raise ValueError(f"edits[{edit_index}].oldText must not be empty in {display_path}")

        occurrences = count_occurrences(original_content, old_text)
        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of edits[{edit_index}] in {display_path}; "
                "oldText must be unique"
            )
        if occurrences == 1:
            start = original_content.index(old_text)
            length = len(old_text)
        else:
            fuzzy_match = _find_unique_fuzzy_match(original_content, old_text)
            if fuzzy_match is None:
                raise ValueError(f"Could not find edits[{edit_index}] in {display_path}")
            start, end = fuzzy_match
            length = end - start

        matched.append(MatchedEdit(edit_index, start, length, new_text))

    matched.sort(key=lambda item: item.start)
    for previous, current in zip(matched, matched[1:]):
        if previous.start + previous.length > current.start:
            raise ValueError(
                f"edits[{previous.index}] and edits[{current.index}] overlap in {display_path}"
            )

    result = original_content
    for edit in reversed(matched):
        result = result[: edit.start] + edit.new_text + result[edit.start + edit.length :]
    if result == original_content:
        raise ValueError(f"No changes made to {display_path}")
    return result


def generate_edit_result(
    display_path: str,
    original_content: str,
    new_content: str,
    replacement_count: int,
) -> EditResult:
    original_lines = original_content.splitlines()
    new_lines = new_content.splitlines()
    matcher = difflib.SequenceMatcher(a=original_lines, b=new_lines, autojunk=False)
    first_changed_line = 1
    for tag, original_start, _, _, _ in matcher.get_opcodes():
        if tag != "equal":
            first_changed_line = original_start + 1
            break

    patch_lines = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"a/{display_path}",
        tofile=f"b/{display_path}",
        lineterm="",
    )
    patch = "\n".join(patch_lines)
    diff = "\n".join(
        line
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    return EditResult(replacement_count, first_changed_line, diff, patch)


def edit_file(
    path: Path,
    display_path: str,
    edits: Sequence[ReplacementInput | Mapping[str, str]],
) -> EditResult:
    """Execute a Pi-style edit while preserving BOM and line endings."""
    with file_mutation_lock(path):
        if not path.is_file():
            raise FileNotFoundError(f"Could not edit file: {display_path}. File does not exist")
        if not os.access(path, os.R_OK | os.W_OK):
            raise PermissionError(f"Could not edit file: {display_path}. File is not readable and writable")

        raw_content = path.read_bytes().decode("utf-8")
        bom = "\ufeff" if raw_content.startswith("\ufeff") else ""
        content = raw_content[len(bom) :]
        line_ending = detect_line_ending(content)
        normalized_content = normalize_to_lf(content)
        new_content = apply_edits_to_content(normalized_content, edits, display_path)
        final_content = bom + restore_line_endings(new_content, line_ending)
        path.write_bytes(final_content.encode("utf-8"))
        return generate_edit_result(display_path, normalized_content, new_content, len(edits))
