# NOVA Coding Agent

A transparent Python coding agent with workspace-aware tools and a LangGraph tool-calling loop.

## Included tools

- `bash(command, timeout)`: runs Bash in the selected workspace and returns stdout, stderr, and the exit code.
- `read(path, offset, limit)`: reads UTF-8 text with one-based line numbers and optional pagination.
- `search(query, path, file_glob, max_results)`: finds literal text with workspace-relative paths and line numbers.
- `find_files(pattern, path, max_results)`: discovers workspace files using glob patterns.
- `git_status()`: reports the current branch and concise working-tree state.
- `git_diff(path, staged, max_chars)`: returns a bounded staged or unstaged diff.
- `write(path, content)`: creates or completely overwrites a file, including missing parent directories.
- `edit(path, edits)`: applies one or more unique, non-overlapping replacements against the same original file.
- `delete(path, recursive)`: deletes files, symlinks, or directories; non-empty directories require `recursive=true`.

Filesystem tools reject paths that escape the selected workspace. `bash` is intentionally unrestricted and has the same operating-system permissions as the Python process. Use a container or sandbox when commands require stronger isolation.

## Agent loop

`python_agent/agent.py` builds this graph:

```text
START -> agent -> tools -> agent -> ...
                   |
                   +-> END when the model returns no tool calls
```

The model is bound to all nine tools. `tools_condition` routes tool-calling responses to `ToolNode`; tool results are appended to `MessagesState`, then the model runs again. The loop stops when the model returns a normal assistant response.

Expected operational failures such as ambiguous replacements, missing files, and overlapping edits are returned to the model as tool messages instead of terminating the graph. The model can inspect the error, read the current file state, and retry with unique context.

To prevent retry loops, NOVA stops after three failures with identical tool arguments or five consecutive tool failures. Live CLI output includes the bounded exception message so failed edit context can be diagnosed instead of showing only `ValueError`.

The `edit` tool matches every `oldText` against the same original content, rejects empty/missing/duplicate/overlapping/no-op edits, applies replacements from right to left, preserves UTF-8 BOM and LF/CRLF line endings, serializes concurrent mutations per file, and returns a unified diff with the first changed line. It also repairs JSON-string and legacy argument forms and conservatively matches common model differences in trailing whitespace, smart quotes, Unicode dashes, and special spaces.

## Setup

Python 3.11 or newer is required.

```bash
cd /path/to/nova-agent
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Set an OpenAI API key without committing it:

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5-mini'  # optional
```

## Run

Interactive mode, with tools restricted to the current directory for filesystem operations:

```bash
.venv/bin/python -m python_agent.cli --workspace "$PWD"
```

Type `@` in interactive mode to autocomplete a workspace file reference:

```text
> Explain @python_agent/tools.py
> Update @"docs/design notes.md" and run its tests
```

Small tagged text files are attached to the prompt automatically. Large, binary, missing, outside-workspace, and sensitive files such as `.env` are represented by metadata only. File suggestions exclude generated dependency trees and sensitive filenames.

Every request prints live tool activity and elapsed time, followed by total prompt time:

```text
[tool] search(query='WorkspaceTools', path='python_agent', file_glob='*.py') started
[tool] search completed in 0.012s
[prompt] completed in 1.438s; 1 tool call: search
```

Assistant text is streamed token-by-token as the model generates it. Tool calls remain live and line-oriented; if a tool starts while an assistant line is open, the CLI moves the tool status to a new line. The final graph state is still collected so interactive conversation history works exactly as before.

Large edit payloads are summarized rather than printed (for example, `content=<500 chars>`), and arguments whose names indicate credentials or secrets are redacted.

One prompt and exit:

```bash
.venv/bin/python -m python_agent.cli \
  --workspace /path/to/project \
  "Read the project and create a short architecture summary"
```

Installing the project also provides the equivalent `python-coding-agent` command.

## Embed the graph

```python
from langchain_openai import ChatOpenAI
from python_agent.agent import create_agent

model = ChatOpenAI(model="gpt-5-mini", temperature=0)
agent = create_agent(model, workspace="/path/to/project")
result = agent.invoke({"messages": [("user", "Create hello.txt containing hello")]})
print(result["messages"][-1].content)
```

Any `BaseChatModel` implementation that supports `bind_tools()` can replace `ChatOpenAI`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests execute real filesystem and Bash operations in temporary directories and use a deterministic fake tool-calling model for the complete graph loop. They do not call a paid model API.
