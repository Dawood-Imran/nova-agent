from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence, cast
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from prompt_toolkit import PromptSession

from .agent import create_agent
from .file_references import WorkspaceFileCompleter, build_referenced_file_context

load_dotenv()  # Load environment variables from .env file


class ConsoleStream:
    """Coordinate streamed assistant tokens with line-oriented tool status output."""

    def __init__(
        self,
        text_output: Callable[[str], None] | None = None,
        line_output: Callable[[str], None] | None = None,
    ) -> None:
        self.text_output = text_output or (lambda text: print(text, end="", flush=True))
        self.line_output = line_output or print
        self.received_text = False
        self._line_open = False

    def token(self, text: str) -> None:
        if not text:
            return
        self.text_output(text)
        self.received_text = True
        self._line_open = not text.endswith(("\n", "\r"))

    def status(self, message: str) -> None:
        if self._line_open:
            self.line_output("")
        self.line_output(message)
        self._line_open = False

    def finish(self) -> None:
        if self._line_open:
            self.line_output("")
            self._line_open = False


class ToolUsageTracker(BaseCallbackHandler):
    """Print live tool activity and retain a concise per-prompt usage summary."""

    summarized_arguments = {"content"}
    sensitive_argument_fragments = {"api_key", "password", "secret", "token"}

    def __init__(
        self,
        output: Callable[[str], None] = print,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.output = output
        self.clock = clock
        self._running: dict[UUID, tuple[str, float]] = {}
        self.tool_names: list[str] = []

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        name = serialized.get("name") or "tool"
        self._running[run_id] = (name, self.clock())
        call = self._format_tool_call(name, inputs, input_str)
        self.output(f"[tool] {call} started")

    def _format_tool_call(
        self,
        name: str,
        inputs: dict[str, Any] | None,
        input_str: str,
    ) -> str:
        if inputs:
            arguments = ", ".join(
                f"{key}={self._format_argument(key, value)}" for key, value in inputs.items()
            )
            return f"{name}({arguments})"
        if input_str:
            preview = input_str if len(input_str) <= 160 else f"{input_str[:157]}..."
            return f"{name}({preview})"
        return name

    def _format_argument(self, key: str, value: Any) -> str:
        lowered_key = key.lower()
        if any(fragment in lowered_key for fragment in self.sensitive_argument_fragments):
            return "<redacted>"
        if key == "edits" and isinstance(value, list):
            noun = "replacement" if len(value) == 1 else "replacements"
            return f"<{len(value)} {noun}>"
        if key in self.summarized_arguments and isinstance(value, str):
            return f"<{len(value)} chars>"
        if isinstance(value, str) and len(value) > 120:
            return repr(f"{value[:117]}...")
        return repr(value)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        del output, kwargs
        finished = self.clock()
        name, started = self._running.pop(run_id, ("tool", finished))
        elapsed = finished - started
        self.tool_names.append(name)
        self.output(f"[tool] {name} completed in {elapsed:.3f}s")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        finished = self.clock()
        name, started = self._running.pop(run_id, ("tool", finished))
        elapsed = finished - started
        self.tool_names.append(name)
        detail = " ".join(str(error).split())
        if len(detail) > 200:
            detail = f"{detail[:197]}..."
        suffix = f": {detail}" if detail else ""
        self.output(f"[tool] {name} failed in {elapsed:.3f}s: {type(error).__name__}{suffix}")

    def finish_prompt(self, elapsed: float) -> None:
        count = len(self.tool_names)
        noun = "tool call" if count == 1 else "tool calls"
        names = ", ".join(self.tool_names) if self.tool_names else "none"
        self.output(f"[prompt] completed in {elapsed:.3f}s; {count} {noun}: {names}")


def prepare_prompt(prompt: str, workspace: str | Path) -> str:
    """Attach bounded content for files explicitly referenced with @."""
    context = build_referenced_file_context(workspace, prompt)
    return f"{prompt}\n\n{context}" if context else prompt


def _streamed_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def run_prompt(
    agent: CompiledStateGraph,
    history: Sequence[BaseMessage],
    prompt: str,
    workspace: str | Path | None = None,
    tracker: ToolUsageTracker | None = None,
    token_output: Callable[[str], None] | None = None,
) -> tuple[list[BaseMessage], str]:
    """Run one user turn and return the complete graph history and final text."""
    prepared_prompt = prepare_prompt(prompt, workspace) if workspace is not None else prompt
    started = perf_counter()
    try:
        config: RunnableConfig | None = {"callbacks": [tracker]} if tracker is not None else None
        graph_input = {"messages": [*history, HumanMessage(content=prepared_prompt)]}
        emitted_text = False
        if token_output is None:
            result = cast(dict[str, Any], agent.invoke(graph_input, config=config))
        else:
            result: dict[str, Any] | None = None
            for mode, payload in agent.stream(
                graph_input,
                config=config,
                stream_mode=["messages", "values"],
            ):
                if mode == "messages":
                    chunk, _metadata = payload
                    if isinstance(chunk, AIMessageChunk):
                        text_chunk = _streamed_text(chunk.content)
                        if text_chunk:
                            token_output(text_chunk)
                            emitted_text = True
                elif mode == "values":
                    result = cast(dict[str, Any], payload)
            if result is None:
                raise RuntimeError("Agent stream ended without a final state")

        messages = cast(list[BaseMessage], list(result["messages"]))
        if prepared_prompt != prompt and len(messages) > len(history):
            messages[len(history)] = HumanMessage(content=prompt)
        final_content = messages[-1].content
        text = final_content if isinstance(final_content, str) else str(final_content)
        if token_output is not None and not emitted_text and text:
            token_output(text)
        return messages, text
    finally:
        if tracker is not None:
            tracker.finish_prompt(perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Python LangGraph coding agent")
    parser.add_argument("prompt", nargs="?", help="Run one prompt and exit; omit for an interactive session")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace available to the tools")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        help="OpenAI model name (default: OPENAI_MODEL or gpt-5-mini)",
    )
    args = parser.parse_args()

    model = ChatOpenAI(model=args.model, temperature=0, streaming=True)
    agent = create_agent(model, args.workspace)
    history: list[BaseMessage] = []

    if args.prompt:
        console = ConsoleStream()
        tracker = ToolUsageTracker(output=console.status)
        _, text = run_prompt(
            agent,
            history,
            args.prompt,
            args.workspace,
            tracker,
            console.token,
        )
        console.finish()
        if not console.received_text and text:
            print(text)
        return

    print(f"Workspace: {args.workspace.resolve()}")
    print("Enter a request, use @ to reference a file, or type exit to quit.")
    prompt_session: PromptSession[str] = PromptSession(
        completer=WorkspaceFileCompleter(args.workspace),
        complete_while_typing=True,
    )
    while True:
        try:
            prompt = prompt_session.prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"exit", "quit"}:
            return
        if not prompt:
            continue
        console = ConsoleStream()
        tracker = ToolUsageTracker(output=console.status)
        history, text = run_prompt(
            agent,
            history,
            prompt,
            args.workspace,
            tracker,
            console.token,
        )
        console.finish()
        if not console.received_text and text:
            print(text)


if __name__ == "__main__":
    main()
