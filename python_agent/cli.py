from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from prompt_toolkit import PromptSession

from .agent import create_agent
from .file_references import WorkspaceFileCompleter, build_referenced_file_context

load_dotenv()  # Load environment variables from .env file


class ToolUsageTracker(BaseCallbackHandler):
    """Print live tool activity and retain a concise per-prompt usage summary."""

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
        **kwargs: Any,
    ) -> None:
        del input_str, kwargs
        name = serialized.get("name") or "tool"
        self._running[run_id] = (name, self.clock())
        self.output(f"[tool] {name} started")

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
        self.output(f"[tool] {name} failed in {elapsed:.3f}s: {type(error).__name__}")

    def finish_prompt(self, elapsed: float) -> None:
        count = len(self.tool_names)
        noun = "tool call" if count == 1 else "tool calls"
        names = ", ".join(self.tool_names) if self.tool_names else "none"
        self.output(f"[prompt] completed in {elapsed:.3f}s; {count} {noun}: {names}")


def prepare_prompt(prompt: str, workspace: str | Path) -> str:
    """Attach bounded content for files explicitly referenced with @."""
    context = build_referenced_file_context(workspace, prompt)
    return f"{prompt}\n\n{context}" if context else prompt


def run_prompt(
    agent: CompiledStateGraph,
    history: Sequence[BaseMessage],
    prompt: str,
    workspace: str | Path | None = None,
    tracker: ToolUsageTracker | None = None,
) -> tuple[list[BaseMessage], str]:
    """Run one user turn and return the complete graph history and final text."""
    prepared_prompt = prepare_prompt(prompt, workspace) if workspace is not None else prompt
    started = perf_counter()
    try:
        config = {"callbacks": [tracker]} if tracker is not None else None
        result = agent.invoke(
            {"messages": [*history, HumanMessage(content=prepared_prompt)]},
            config=config,
        )
    finally:
        if tracker is not None:
            tracker.finish_prompt(perf_counter() - started)
    messages = list(result["messages"])
    if prepared_prompt != prompt and len(messages) > len(history):
        messages[len(history)] = HumanMessage(content=prompt)
    final_content = messages[-1].content
    text = final_content if isinstance(final_content, str) else str(final_content)
    return messages, text


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

    model = ChatOpenAI(model=args.model, temperature=0)
    agent = create_agent(model, args.workspace)
    history: list[BaseMessage] = []

    if args.prompt:
        tracker = ToolUsageTracker()
        _, text = run_prompt(agent, history, args.prompt, args.workspace, tracker)
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
        tracker = ToolUsageTracker()
        history, text = run_prompt(agent, history, prompt, args.workspace, tracker)
        print(text)


if __name__ == "__main__":
    main()
