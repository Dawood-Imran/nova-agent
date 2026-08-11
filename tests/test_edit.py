import json
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from python_agent.tools import WorkspaceTools, build_tools


def test_edit_applies_multiple_non_overlapping_replacements_against_original(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write(
        "sample.py",
        "if __name__ == '__main__':\n"
        "    print('example')\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n",
    )

    result = tools.edit(
        "sample.py",
        [
            {
                "oldText": "\ndef subtract(a, b):\n    return a - b\n",
                "newText": "",
            },
            {
                "oldText": "if __name__ == '__main__':",
                "newText": "def subtract(a, b):\n    return a - b\n\nif __name__ == '__main__':",
            },
        ],
    )

    content = (tmp_path / "sample.py").read_text(encoding="utf-8")
    assert content.count("def subtract") == 1
    assert content.startswith("def subtract")
    assert "Successfully replaced 2 block(s) in sample.py." in result
    assert "First changed line: 1" in result
    assert "--- a/sample.py" in result
    assert "+++ b/sample.py" in result


def test_edit_rejects_missing_duplicate_overlapping_and_noop_edits(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write("values.txt", "same\nsame\nlast\n")

    with pytest.raises(ValueError, match="Found 2 occurrences"):
        tools.edit("values.txt", [{"oldText": "same", "newText": "new"}])
    with pytest.raises(ValueError, match="Could not find"):
        tools.edit("values.txt", [{"oldText": "missing", "newText": "new"}])
    with pytest.raises(ValueError, match="must not be empty"):
        tools.edit("values.txt", [{"oldText": "", "newText": "new"}])
    with pytest.raises(ValueError, match="overlap"):
        tools.edit(
            "values.txt",
            [
                {"oldText": "same\nsame", "newText": "first"},
                {"oldText": "same\nlast", "newText": "second"},
            ],
        )
    with pytest.raises(ValueError, match="No changes made"):
        tools.edit("values.txt", [{"oldText": "last", "newText": "last"}])

    assert (tmp_path / "values.txt").read_text(encoding="utf-8") == "same\nsame\nlast\n"


def test_edit_preserves_utf8_bom_and_crlf_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "windows.py"
    target.write_bytes("\ufeffflag = False\r\nprint(flag)\r\n".encode("utf-8"))
    tools = WorkspaceTools(tmp_path)

    tools.edit("windows.py", [{"oldText": "flag = False\n", "newText": "flag = True\n"}])

    assert target.read_bytes() == "\ufeffflag = True\r\nprint(flag)\r\n".encode("utf-8")


def test_edit_fuzzy_matches_common_model_unicode_and_whitespace_differences(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write("message.txt", "message = “hello”—world   \n")

    tools.edit(
        "message.txt",
        [{"oldText": 'message = "hello"-world\n', "newText": 'message = "updated"\n'}],
    )

    assert (tmp_path / "message.txt").read_text(encoding="utf-8") == 'message = "updated"\n'


def test_edit_tool_repairs_json_string_and_legacy_arguments(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in build_tools(tmp_path)}
    (tmp_path / "config.txt").write_text("enabled=false\nretries=2\n", encoding="utf-8")

    result = tools["edit"].invoke(
        {
            "path": "config.txt",
            "edits": json.dumps([{"oldText": "enabled=false", "newText": "enabled=true"}]),
            "oldText": "retries=2",
            "newText": "retries=3",
        }
    )

    assert "Successfully replaced 2 block(s)" in result
    assert (tmp_path / "config.txt").read_text(encoding="utf-8") == "enabled=true\nretries=3\n"


def test_edit_tool_rejects_incomplete_legacy_arguments(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in build_tools(tmp_path)}
    (tmp_path / "config.txt").write_text("enabled=false\n", encoding="utf-8")

    with pytest.raises(ValueError):
        tools["edit"].invoke(
            {
                "path": "config.txt",
                "edits": [{"oldText": "enabled=false", "newText": "enabled=true"}],
                "oldText": "incomplete legacy argument",
            }
        )

    assert (tmp_path / "config.txt").read_text(encoding="utf-8") == "enabled=false\n"


def test_edit_tool_returns_concise_model_content_and_diff_artifact(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in build_tools(tmp_path)}
    (tmp_path / "config.txt").write_text("enabled=false\n", encoding="utf-8")

    message = tools["edit"].invoke(
        {
            "name": "edit",
            "args": {
                "path": "config.txt",
                "edits": [{"oldText": "enabled=false", "newText": "enabled=true"}],
            },
            "id": "edit-with-artifact",
            "type": "tool_call",
        }
    )

    assert isinstance(message, ToolMessage)
    assert message.content == "Successfully replaced 1 block(s) in config.txt."
    assert message.artifact["firstChangedLine"] == 1
    assert "-enabled=false" in message.artifact["diff"]
    assert "--- a/config.txt" in message.artifact["patch"]
