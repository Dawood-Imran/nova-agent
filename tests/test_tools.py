from pathlib import Path
import subprocess

import pytest

from python_agent.tools import WorkspaceTools, build_tools


def init_git_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "nova@example.test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "NOVA Tests"], cwd=path, check=True)


def test_write_read_edit_delete_file(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)

    write_result = tools.write("notes/example.txt", "first\nsecond\nthird\n")
    # Verify that a second write without an overwrite flag fails if the file exists
    with pytest.raises(ValueError, match="already exists"):
        tools.write("notes/example.txt", "new content")
    # Verify that write with overwrite=True succeeds
    overwrite_result = tools.write("notes/example.txt", "overwritten", overwrite=True)
    assert "overwrote" in overwrite_result.lower()
    assert (tmp_path / "notes/example.txt").read_text() == "overwritten"

    assert write_result == "Wrote 19 characters to notes/example.txt"
    # The file was overwritten to "overwritten", which is 1 line.
    # Offset 2 should now be EOF.
    assert "end of file" in tools.read("notes/example.txt", offset=2).lower()

    # Restore content for edit test
    tools.write("notes/example.txt", "first\nsecond\nthird\n", overwrite=True)
    edit_result = tools.edit(
        "notes/example.txt",
        [{"oldText": "second", "newText": "changed"}],
    )
    assert "Successfully replaced 1 block(s)" in edit_result
    assert (tmp_path / "notes/example.txt").read_text() == "first\nchanged\nthird\n"

    assert tools.delete("notes/example.txt") == "Deleted file notes/example.txt"
    assert not (tmp_path / "notes/example.txt").exists()



def test_paths_cannot_escape_workspace(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tools.read("../secret.txt")


def test_delete_directory_requires_recursive_flag(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write("nested/file.txt", "data")

    with pytest.raises(ValueError, match="recursive=True"):
        tools.delete("nested")

    assert tools.delete("nested", recursive=True) == "Deleted directory nested"
    assert not (tmp_path / "nested").exists()


def test_delete_removes_symlink_without_deleting_target(tmp_path: Path) -> None:
    outside_file = tmp_path.parent / "outside-target.txt"
    outside_file.write_text("keep me")
    link = tmp_path / "target-link"
    link.symlink_to(outside_file)
    tools = WorkspaceTools(tmp_path)

    assert tools.delete("target-link") == "Deleted file target-link"
    assert not link.exists()
    assert outside_file.read_text() == "keep me"


def test_bash_runs_in_workspace_and_reports_failures(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)

    success = tools.bash("pwd && printf hello")
    assert f"stdout:\n{tmp_path}\nhello" in success
    assert "exit_code: 0" in success

    failure = tools.bash("printf problem >&2; exit 7")
    assert "stderr:\nproblem" in failure
    assert "exit_code: 7" in failure


def test_bash_timeout_terminates_command(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)

    result = tools.bash("sleep 2", timeout=0.01)

def test_read_eof_returns_empty_with_metadata(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    file_path = tmp_path / "short.txt"
    file_path.write_text("line1\nline2", encoding="utf-8")
    # Offset beyond EOF should return an empty string and a message rather than raising ValueError
    result = tools.read("short.txt", offset=10)
    assert "end of file" in result.lower()


def test_build_tools_exposes_langchain_tools(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in build_tools(tmp_path)}

    assert set(tools) == {
        "bash",
        "delete",
        "edit",
        "find_files",
        "git_diff",
        "git_status",
        "read",
        "search",
        "write",
    }
    assert tools["write"].invoke({"path": "from-tool.txt", "content": "hello"}) == (
        "Wrote 5 characters to from-tool.txt"
    )
    assert tools["read"].invoke({"path": "from-tool.txt"}) == "1|hello"


def test_search_finds_literal_text_with_line_numbers_and_glob(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write("src/app.py", "first\nflag = False\n")
    tools.write("src/other.py", "flag = False\n")
    tools.write("src/notes.txt", "flag = False\n")

    result = tools.search("flag = False", path="src", file_glob="*.py")

    assert result == "src/app.py:2:flag = False\nsrc/other.py:1:flag = False"


def test_search_is_bounded_and_rejects_workspace_escape(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write("matches.txt", "match\nmatch\nmatch\n")

    assert tools.search("match", max_results=2) == (
        "matches.txt:1:match\nmatches.txt:2:match\n[results truncated at 2 matches]"
    )
    with pytest.raises(ValueError, match="outside the workspace"):
        tools.search("secret", path="..")


def test_find_files_returns_filtered_workspace_paths(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write("src/app.py", "")
    tools.write("src/nested/helper.py", "")
    tools.write("src/notes.txt", "")

    assert tools.find_files("**/*.py", path="src") == [
        "src/app.py",
        "src/nested/helper.py",
    ]


def test_search_does_not_follow_file_symlinks_outside_workspace(tmp_path: Path) -> None:
    outside_file = tmp_path.parent / "outside-search-target.txt"
    outside_file.write_text("private marker", encoding="utf-8")
    (tmp_path / "outside-link.txt").symlink_to(outside_file)
    tools = WorkspaceTools(tmp_path)

    assert tools.search("private marker") == "(no matches)"


def test_git_status_reports_branch_and_worktree_changes(tmp_path: Path) -> None:
    init_git_repository(tmp_path)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True)
    tracked.write_text("after\n", encoding="utf-8")
    tools = WorkspaceTools(tmp_path)

    status = tools.git_status()

    assert "## main" in status
    assert " M tracked.txt" in status


def test_git_diff_supports_path_staged_and_bounded_output(tmp_path: Path) -> None:
    init_git_repository(tmp_path)
    tracked = tmp_path / "tracked.txt"
    other = tmp_path / "other.txt"
    tracked.write_text("before\n", encoding="utf-8")
    other.write_text("unchanged\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True)
    tracked.write_text("after\n", encoding="utf-8")
    tools = WorkspaceTools(tmp_path)

    unstaged = tools.git_diff(path="tracked.txt")
    assert "-before" in unstaged
    assert "+after" in unstaged
    assert "other.txt" not in unstaged

    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    assert tools.git_diff(path="tracked.txt") == "(no diff)"
    assert "+after" in tools.git_diff(path="tracked.txt", staged=True)
    assert tools.git_diff(path="tracked.txt", staged=True, max_chars=20).endswith(
        "\n[diff truncated at 20 characters]"
    )
