from pathlib import Path
from uuid import uuid4

from python_agent.cli import ToolUsageTracker, prepare_prompt


def test_prepare_prompt_adds_tagged_file_context(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("flag = False\n", encoding="utf-8")

    prepared = prepare_prompt("Explain @app.py", tmp_path)

    assert prepared.startswith("Explain @app.py\n\nExplicitly referenced workspace files:")
    assert "flag = False" in prepared


def test_prepare_prompt_leaves_prompt_unchanged_without_references(tmp_path: Path) -> None:
    assert prepare_prompt("Explain the project", tmp_path) == "Explain the project"


def test_tool_usage_tracker_reports_tool_and_prompt_timings() -> None:
    output: list[str] = []
    times = iter([10.0, 10.25])
    tracker = ToolUsageTracker(output=output.append, clock=lambda: next(times))
    run_id = uuid4()

    tracker.on_tool_start({"name": "read"}, "", run_id=run_id)
    tracker.on_tool_end("contents", run_id=run_id)
    tracker.finish_prompt(1.5)

    assert output == [
        "[tool] read started",
        "[tool] read completed in 0.250s",
        "[prompt] completed in 1.500s; 1 tool call: read",
    ]
    assert tracker.tool_names == ["read"]


def test_tool_usage_tracker_reports_tool_errors() -> None:
    output: list[str] = []
    times = iter([5.0, 5.1])
    tracker = ToolUsageTracker(output=output.append, clock=lambda: next(times))
    run_id = uuid4()

    tracker.on_tool_start({"name": "update"}, "", run_id=run_id)
    tracker.on_tool_error(ValueError("failed"), run_id=run_id)

    assert output[-1] == "[tool] update failed in 0.100s: ValueError"
