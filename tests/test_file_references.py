from pathlib import Path

from prompt_toolkit.document import Document

from python_agent.file_references import (
    WorkspaceFileCompleter,
    build_referenced_file_context,
    extract_file_references,
)


def test_extract_file_references_supports_plain_quoted_and_duplicate_paths() -> None:
    prompt = 'Compare @src/app.py with @"docs/design notes.md" and @src/app.py'

    assert extract_file_references(prompt) == ["src/app.py", "docs/design notes.md"]


def test_extract_file_references_does_not_treat_email_as_a_file() -> None:
    assert extract_file_references("Email user@example.com about @src/app.py") == ["src/app.py"]


def test_build_referenced_file_context_includes_small_text_files(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("flag = False\n", encoding="utf-8")

    context = build_referenced_file_context(tmp_path, "Explain @src/app.py")

    assert '<referenced_file path="src/app.py" content_included="true">' in context
    assert "flag = False" in context


def test_build_referenced_file_context_uses_metadata_for_large_files(tmp_path: Path) -> None:
    source = tmp_path / "large.py"
    source.write_text("x" * 100, encoding="utf-8")

    context = build_referenced_file_context(tmp_path, "Explain @large.py", max_file_bytes=20)

    assert 'path="large.py" content_included="false"' in context
    assert "reason=\"file_too_large\"" in context
    assert "x" * 100 not in context


def test_build_referenced_file_context_does_not_expose_sensitive_files(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret", encoding="utf-8")

    context = build_referenced_file_context(tmp_path, "Read @.env")

    assert 'path=".env" content_included="false"' in context
    assert 'reason="sensitive_file"' in context
    assert "secret" not in context


def test_build_referenced_file_context_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    context = build_referenced_file_context(tmp_path, "Read @../secret.txt")

    assert 'path="../secret.txt" content_included="false"' in context
    assert 'reason="outside_workspace"' in context


def test_workspace_file_completer_suggests_matching_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "python_agent" / "tools.py"
    source.parent.mkdir()
    source.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("secret", encoding="utf-8")
    completer = WorkspaceFileCompleter(tmp_path)

    completions = list(
        completer.get_completions(Document("Update @python_agent/to"), complete_event=None)
    )

    assert [(item.text, item.start_position) for item in completions] == [
        ("python_agent/tools.py", -len("python_agent/to"))
    ]


def test_workspace_file_completer_quotes_paths_with_spaces(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "design notes.md"
    source.parent.mkdir()
    source.write_text("", encoding="utf-8")
    completer = WorkspaceFileCompleter(tmp_path)

    completions = list(completer.get_completions(Document("Read @des"), complete_event=None))

    assert [(item.text, item.start_position) for item in completions] == [
        ('"docs/design notes.md"', -len("des"))
    ]
