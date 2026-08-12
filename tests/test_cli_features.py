from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from python_agent.cli import (
    ConsoleStream,
    ToolUsageTracker,
    build_intervention_prompt,
    prepare_prompt,
    run_prompt,
)


class FakeStreamingAgent:
    def stream(self, input, config=None, stream_mode=None):
        del config
        assert stream_mode == ["messages", "values"]
        yield "messages", (AIMessageChunk(content="Hel"), {"langgraph_node": "agent"})
        yield "messages", (AIMessageChunk(content="lo"), {"langgraph_node": "agent"})
        yield "values", {"messages": [*input["messages"], AIMessage(content="Hello")]}


class FakeInterruptingAgent:
    def stream(self, input, config=None, stream_mode=None):
        del input, config, stream_mode
        raise KeyboardInterrupt


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

    tracker.on_tool_start({"name": "edit"}, "", run_id=run_id)
    tracker.on_tool_error(ValueError("failed"), run_id=run_id)

    assert output[-1] == "[tool] edit failed in 0.100s: ValueError: failed"


def test_tool_usage_tracker_displays_concise_tool_arguments() -> None:
    output: list[str] = []
    tracker = ToolUsageTracker(output=output.append, clock=lambda: 1.0)

    tracker.on_tool_start(
        {"name": "search"},
        "",
        run_id=uuid4(),
        inputs={"query": "WorkspaceTools", "path": "python_agent", "file_glob": "*.py"},
    )
    tracker.on_tool_start(
        {"name": "edit"},
        "",
        run_id=uuid4(),
        inputs={
            "path": "app.py",
            "edits": [{"oldText": "flag = False", "newText": "flag = True"}],
        },
    )

    assert output == [
        "[tool] search(query='WorkspaceTools', path='python_agent', file_glob='*.py') started",
        "[tool] edit(path='app.py', edits=<1 replacement>) started",
    ]


def test_run_prompt_streams_tokens_and_returns_final_history() -> None:
    tokens: list[str] = []

    history, text = run_prompt(
        FakeStreamingAgent(),
        [],
        "Hi",
        token_output=tokens.append,
    )

    assert tokens == ["Hel", "lo"]
    assert isinstance(history[0], HumanMessage)
    assert history[-1].content == "Hello"
    assert text == "Hello"


def test_run_prompt_interruption_keeps_existing_history() -> None:
    original_history = [HumanMessage(content="previous")]
    output: list[str] = []
    tracker = ToolUsageTracker(output=output.append)

    history, text = run_prompt(
        FakeInterruptingAgent(),
        original_history,
        "Stop this",
        tracker=tracker,
        token_output=lambda token: None,
    )

    assert history == original_history
    assert text is None
    assert output[0] == "\n[interrupt] User pressed Esc/Ctrl-C. Stopping current turn..."


def test_console_stream_separates_tokens_from_status_lines() -> None:
    output: list[str] = []
    stream = ConsoleStream(
        text_output=lambda text: output.append(f"text:{text}"),
        line_output=lambda text="": output.append(f"line:{text}"),
    )

    stream.token("Hello")
    stream.status("[tool] read started")
    stream.token("Done\n")
    stream.finish()

    assert output == [
        "text:Hello",
        "line:",
        "line:[tool] read started",
        "text:Done\n",
    ]
    assert stream.received_text is True


def test_console_stream_default_token_output_does_not_crash(capsys) -> None:
    stream = ConsoleStream()

    stream.token("hey")
    stream.finish()

    captured = capsys.readouterr()
    assert "hey" in captured.out


def test_console_stream_default_status_renders_tool_dashboard(capsys) -> None:
    stream = ConsoleStream()

    stream.status("[tool] read(path='app.py') started")
    stream.status("[tool] read completed in 0.123s")

    captured = capsys.readouterr()
    assert "Tool Activity" in captured.out
    assert "read" in captured.out
    assert "completed" in captured.out


def test_build_intervention_prompt_includes_original_and_correction() -> None:
    prompt = build_intervention_prompt("make endpoint", "use port 8001 instead")

    assert "Previous turn was interrupted" in prompt
    assert "make endpoint" in prompt
    assert "use port 8001 instead" in prompt
