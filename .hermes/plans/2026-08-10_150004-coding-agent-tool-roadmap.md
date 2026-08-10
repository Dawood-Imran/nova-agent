# Coding Agent Tool Expansion Implementation Plan

> **For Hermes:** Implement this plan sequentially with tests after every tool; do not add all tools in one unverified change.

**Goal:** Expand the coding agent with token-efficient discovery and safer editing tools while keeping every operation workspace-scoped and deterministic.

**Architecture:** Add one capability at a time to `WorkspaceTools`, expose it through `build_tools()`, and update the system prompt only after tests establish the tool contract. Start with discovery tools, then add contextual patching and concurrency protection. Preserve the existing exact `update` tool as the simplest safe replacement operation.

**Tech Stack:** Python 3.11+, pathlib, regex/glob standard library support, LangChain `@tool`, LangGraph `ToolNode`, pytest.

---

## Recommended order and tool descriptions

### 1. `search` — search inside files

**Priority:** First

**LLM-facing description:**

> Search for literal text inside workspace files. Return matching file paths, one-based line numbers, and short line snippets. Optionally restrict files with a glob and cap the result count. Use this before `read` to locate relevant code without loading entire files.

**Suggested inputs:**
- `query: str` — literal text to find
- `path: str = "."` — workspace-relative file or directory
- `file_glob: str | None = None` — for example `"*.py"`
- `max_results: int = 50`

**Why first:** It directly addresses the current token-efficiency problem. The model can find both `flag = "False"` occurrences and then read only the relevant ranges.

**Safety/behavior:**
- Reject paths outside the workspace.
- Search text files only; skip undecodable/binary files.
- Return deterministic ordering.
- Report truncation when matches exceed `max_results`.
- Do not use unrestricted shell commands internally.

### 2. `find_files` — discover files by name or glob

**Priority:** Second

**LLM-facing description:**

> Find files and directories inside the workspace by glob pattern. Return workspace-relative paths in deterministic order. Use this to locate likely files before reading or searching their contents.

**Suggested inputs:**
- `pattern: str` — for example `"**/*.py"` or `"*config*"`
- `path: str = "."`
- `max_results: int = 100`

**Why:** The model currently relies on `bash` to discover the project structure. A bounded, workspace-safe discovery tool is easier for the model and safer than shelling out.

### 3. `apply_patch` — contextual multi-location edits

**Priority:** Third

**LLM-facing description:**

> Apply a contextual unified diff to one or more workspace files. Validate every hunk before writing and make no changes if any hunk cannot be applied. Return a concise diff summary. Use this for insertions, deletions, or several related edits; use `update` for one exact replacement.

**Suggested inputs:**
- `patch: str` — unified diff text

**Why:** It lets the model express small changes without emitting complete files and handles insertions more naturally than exact replacement.

**Safety/behavior:**
- Reject absolute paths and traversal.
- Validate all hunks before modifying anything.
- Apply all changes atomically as a group where practical.
- Preserve unrelated content and line-ending style.
- Return changed paths and hunk counts.

### 4. `file_info` — content version and metadata

**Priority:** Fourth

**LLM-facing description:**

> Return workspace-relative file metadata needed for safe editing: size, line count, newline style, and a SHA-256 content version. Use the version when an edit must fail if the file changed after it was inspected.

**Suggested inputs:**
- `path: str`

**Why:** This provides optimistic concurrency protection and helps avoid overwriting external edits.

### 5. Version-checked edits — harden `update` and `apply_patch`

**Priority:** Fifth; this is an enhancement rather than a separate model tool.

**Contract:** Add an optional `expected_hash` argument. If supplied and the current file hash differs, fail before writing.

**Why:** It closes the gap between the model reading a file and later modifying it.

### 6. `move` — rename or relocate files safely

**Priority:** Sixth

**LLM-facing description:**

> Move or rename a file or directory within the workspace. Refuse to overwrite an existing destination unless explicitly allowed. Use this instead of shell commands for repository file organization.

**Suggested inputs:**
- `source: str`
- `destination: str`
- `overwrite: bool = False`

**Why:** Common coding tasks include renames. A workspace-scoped operation is safer and easier to verify than `mv` through unrestricted Bash.

### 7. `copy` — duplicate files safely

**Priority:** Seventh and optional

**LLM-facing description:**

> Copy a file or directory within the workspace. Refuse to overwrite existing content unless explicitly allowed, and return the copied destination.

**Why:** Useful for templates and migrations, but less important than discovery and patching.

---

## Tools not recommended yet

- A generic `append` tool: placement is usually ambiguous and contextual patching is safer.
- An `occurrence=1` replacement option: silently selecting the first duplicate can edit the wrong scope.
- A line-number-only editor: line numbers become stale; any positional edit should also validate expected text or a file hash.
- AST-specific tools: powerful but language-specific and unnecessary until the generic toolset proves insufficient.
- A dedicated test tool: `bash` already runs tests and builds; first improve discovery and editing.

---

## Sequential implementation tasks

### Task 1: Add `search`

**Files:**
- Modify: `python_agent/tools.py`
- Modify: `tests/test_tools.py`
- Modify: `python_agent/agent.py`
- Modify: `README.md`

**Steps:**
1. Write failing tests for one match, multiple files, glob filtering, result limits, binary skipping, and workspace escape rejection.
2. Run `venv/bin/python -m pytest tests/test_tools.py -q` and confirm the new tests fail.
3. Implement `WorkspaceTools.search()` with literal matching and bounded deterministic output.
4. Expose `search_tool` through `build_tools()`.
5. Update the expected tool names in `tests/test_agent.py`.
6. Update the system prompt to instruct `search` before broad reads.
7. Run `venv/bin/python -m pytest -q`; expected result: all tests pass.

### Task 2: Add `find_files`

**Files:** Same targets as Task 1.

**Steps:** Follow the same red-green-refactor cycle. Cover deterministic ordering, file/directory matching, limits, and workspace boundaries.

### Task 3: Add `apply_patch`

**Files:**
- Modify: `python_agent/tools.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_agent.py`
- Modify: `python_agent/agent.py`
- Modify: `README.md`

**Steps:**
1. Test one-hunk replacement, insertion, deletion, multiple hunks, invalid context, traversal rejection, and all-or-nothing failure.
2. Implement parsing and validation before any write.
3. Use temporary files plus `os.replace()` for atomic per-file writes.
4. Return a concise changed-file/hunk summary.
5. Run the focused and complete test suites.

### Task 4: Add `file_info` and expected hashes

**Files:** Same core and test files.

**Steps:**
1. Test hash stability, hash changes, newline detection, and stale-hash rejection.
2. Implement `file_info`.
3. Add optional `expected_hash` to `update` and patch operations.
4. Verify stale edits fail without changing files.

### Task 5: Add `move`, then optionally `copy`

Implement each as a separate tested change. Cover workspace traversal, symlinks, existing destinations, recursive directories, and overwrite behavior.

---

## Validation after every tool

Run:

```bash
venv/bin/python -m pytest -q
```

Acceptance criteria:
- Existing tests continue to pass.
- Every new tool rejects workspace escapes.
- Tool descriptions clearly tell the model when to use the tool.
- Search/discovery output is bounded and deterministic.
- Editing failures leave files unchanged.
- README examples and the expected bound-tool list remain current.

## Main trade-offs

- More tools improve precision but increase the model's tool-selection burden; keep contracts distinct.
- Literal search is easier and safer initially than regex; regex can be added later if justified.
- A custom patch parser is maintenance-heavy; prefer a well-tested standard-library-compatible strategy or external dependency only after checking project constraints.
- Hash checks improve concurrency safety but require the model to carry the inspected version into its edit request.
