from __future__ import annotations

import os
import shutil
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from .edit import EditToolInput, ReplacementInput, edit_file


class WorkspaceTools:
    """Filesystem and shell operations rooted in one workspace."""

    ignored_directories = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace does not exist or is not a directory: {self.workspace}")

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError(f"Path is outside the workspace: {path}") from error
        return resolved

    def _display(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    def _is_inside_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.workspace)
        except (OSError, ValueError):
            return False
        return True

    def _resolve_entry(self, path: str) -> Path:
        """Resolve a directory entry without following its final symlink."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        if candidate == self.workspace:
            return self.workspace
        resolved = candidate.parent.resolve() / candidate.name
        try:
            resolved.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError(f"Path is outside the workspace: {path}") from error
        return resolved

    def _iter_files(self, path: str = "."):
        """Yield workspace files in deterministic order while skipping generated trees."""
        resolved = self._resolve(path)
        if resolved.is_file():
            yield resolved
            return
        if not resolved.is_dir():
            raise ValueError(f"Path does not exist or is not a directory: {path}")

        files = (
            candidate
            for candidate in resolved.rglob("*")
            if candidate.is_file()
            and self._is_inside_workspace(candidate)
            and not any(
                part in self.ignored_directories
                for part in candidate.relative_to(self.workspace).parts
            )
        )
        yield from sorted(files, key=self._display)

    def find_files(self, pattern: str, path: str = ".", max_results: int = 100) -> list[str]:
        """Find workspace files by glob pattern and return relative paths."""
        if not pattern:
            raise ValueError("pattern must not be empty")
        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        root = self._resolve(path)
        matches = []
        for candidate in self._iter_files(path):
            relative = candidate.relative_to(root) if root.is_dir() else Path(candidate.name)
            matches_pattern = relative.match(pattern)
            if pattern.startswith("**/"):
                matches_pattern = matches_pattern or relative.match(pattern[3:])
            if matches_pattern:
                matches.append(self._display(candidate))
                if len(matches) == max_results:
                    break
        return matches

    def search(
        self,
        query: str,
        path: str = ".",
        file_glob: str | None = None,
        max_results: int = 50,
    ) -> str:
        """Search for literal text and return path, line number, and matching line."""
        if not query:
            raise ValueError("query must not be empty")
        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        matches: list[str] = []
        truncated = False
        for candidate in self._iter_files(path):
            if file_glob and not candidate.match(file_glob):
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if query not in line:
                    continue
                if len(matches) == max_results:
                    truncated = True
                    break
                matches.append(f"{self._display(candidate)}:{line_number}:{line}")
            if truncated:
                break

        if not matches:
            return "(no matches)"
        if truncated:
            matches.append(f"[results truncated at {max_results} matches]")
        return "\n".join(matches)

    def _git(self, arguments: list[str], *, input_text: str | None = None) -> str:
        """Run a non-interactive Git command in the workspace."""
        process = subprocess.run(
            ["git", *arguments],
            cwd=self.workspace,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip() or "unknown Git error"
            raise ValueError(detail)
        return process.stdout


    def git_status(self) -> str:
        """Return the current branch and concise working-tree status."""
        return self._git(["status", "--short", "--branch"]).rstrip() or "(clean)"

    def git_diff(
        self,
        path: str | None = None,
        staged: bool = False,
        max_chars: int = 20_000,
    ) -> str:
        """Return a bounded staged or unstaged Git diff, optionally for one path."""
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")

        arguments = ["diff", "--no-ext-diff", "--no-textconv", "--no-color"]
        if staged:
            arguments.append("--cached")
        if path is not None:
            resolved = self._resolve(path)
            arguments.extend(["--", self._display(resolved)])
        output = self._git(arguments).rstrip()
        if not output:
            return "(no diff)"
        if len(output) > max_chars:
            return f"{output[:max_chars]}\n[diff truncated at {max_chars} characters]"
        return output

    def read(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        """Read UTF-8 text with one-based line numbers and optional pagination."""
        if offset < 1:
            raise ValueError("offset must be at least 1")
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")

        resolved = self._resolve(path)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        if offset > len(lines):
            return f"(end of file: {len(lines)} lines)"
        selected = lines[offset - 1 :] if limit is None else lines[offset - 1 : offset - 1 + limit]
        return "\n".join(f"{number}|{line}" for number, line in enumerate(selected, start=offset))

    def write(self, path: str, content: str, overwrite: bool = False) -> str:
        """Create a UTF-8 text file. Fails if file exists unless overwrite=True."""
        resolved = self._resolve(path)
        if resolved.exists() and not overwrite:
            raise ValueError(f"File already exists: {self._display(resolved)}. Use overwrite=True to replace it.")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        action = "overwrote" if overwrite else "wrote"
        return f"{action.capitalize()} {len(content)} characters to {self._display(resolved)}"

    def edit(
        self,
        path: str,
        edits: Sequence[ReplacementInput | dict[str, str]],
    ) -> str:
        """Apply Pi-style non-overlapping replacements against one original file."""
        content, artifact = self.edit_with_details(path, edits)
        return (
            f"{content}\n"
            f"First changed line: {artifact['firstChangedLine']}\n"
            f"Patch:\n{artifact['patch']}"
        )

    def edit_with_details(
        self,
        path: str,
        edits: Sequence[ReplacementInput | dict[str, str]],
    ) -> tuple[str, dict[str, str | int]]:
        """Execute an edit and separate model-facing text from UI diff details."""
        resolved = self._resolve(path)
        display = self._display(resolved)
        result = edit_file(resolved, display, edits)
        return (
            f"Successfully replaced {result.replacements} block(s) in {display}.",
            {
                "diff": result.diff,
                "patch": result.patch,
                "firstChangedLine": result.first_changed_line,
            },
        )


    def delete(self, path: str, recursive: bool = False) -> str:
        """Delete a file, symlink, empty directory, or recursive directory tree."""
        resolved = self._resolve_entry(path)
        display = self._display(resolved)
        if resolved == self.workspace:
            raise ValueError("Cannot delete the workspace root")
        if not resolved.exists() and not resolved.is_symlink():
            raise FileNotFoundError(f"Path does not exist: {display}")
        if resolved.is_file() or resolved.is_symlink():
            resolved.unlink()
            return f"Deleted file {display}"
        if any(resolved.iterdir()) and not recursive:
            raise ValueError("Directory is not empty; pass recursive=True to delete it")
        if recursive:
            shutil.rmtree(resolved)
        else:
            resolved.rmdir()
        return f"Deleted directory {display}"

    def bash(self, command: str, timeout: float | None = 120.0) -> str:
        """Execute a Bash command in the workspace and capture stdout/stderr."""
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=self.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name != "nt",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate()
            return f"Command timed out after {timeout} seconds"

        sections = []
        if stdout:
            sections.append(f"stdout:\n{stdout.rstrip()}")
        if stderr:
            sections.append(f"stderr:\n{stderr.rstrip()}")
        if not sections:
            sections.append("(no output)")
        sections.append(f"exit_code: {process.returncode}")
        return "\n\n".join(sections)


def build_tools(workspace: str | Path) -> list[BaseTool]:
    """Create LangChain tools bound to a single workspace."""
    operations = WorkspaceTools(workspace)

    @tool("bash")
    def bash_tool(command: str, timeout: float = 120.0) -> str:
        """Execute a Bash command in the workspace and return stdout, stderr, and its exit code."""
        return operations.bash(command, timeout)

    @tool("read")
    def read_tool(path: str, offset: int = 1, limit: int | None = None) -> str:
        """Read a UTF-8 text file with one-based line numbers. Use offset and limit for pagination."""
        return operations.read(path, offset, limit)

    @tool("search")
    def search_tool(
        query: str,
        path: str = ".",
        file_glob: str | None = None,
        max_results: int = 50,
    ) -> str:
        """Search literal text in workspace files. Returns paths, line numbers, and snippets; use before broad reads."""
        return operations.search(query, path, file_glob, max_results)

    @tool("find_files")
    def find_files_tool(pattern: str, path: str = ".", max_results: int = 100) -> list[str]:
        """Find workspace files by glob pattern. Use this to locate files before reading them."""
        return operations.find_files(pattern, path, max_results)

    @tool("git_status")
    def git_status_tool() -> str:
        """Return the current Git branch and concise working-tree status without modifying the repository."""
        return operations.git_status()

    @tool("git_diff")
    def git_diff_tool(
        path: str | None = None,
        staged: bool = False,
        max_chars: int = 20_000,
    ) -> str:
        """Return a bounded staged or unstaged Git diff, optionally limited to one workspace file."""
        return operations.git_diff(path, staged, max_chars)

    @tool("write")
    def write_tool(path: str, content: str, overwrite: bool = False) -> str:
        """Create a UTF-8 text file. Fails if file exists unless overwrite=True."""
        return operations.write(path, content, overwrite=overwrite)

    @tool("edit", args_schema=EditToolInput, response_format="content_and_artifact")
    def edit_tool(
        path: str,
        edits: list[ReplacementInput],
    ) -> tuple[str, dict[str, str | int]]:
        """Apply one or more unique, non-overlapping text replacements matched against the original file."""
        return operations.edit_with_details(path, edits)


    @tool("delete")
    def delete_tool(path: str, recursive: bool = False) -> str:
        """Delete a file or directory. Set recursive only when deleting a non-empty directory tree."""
        return operations.delete(path, recursive)

    return [
        bash_tool,
        read_tool,
        search_tool,
        find_files_tool,
        git_status_tool,
        git_diff_tool,
        write_tool,
        edit_tool,
        delete_tool,
    ]
