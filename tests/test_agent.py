from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from python_agent.agent import create_agent
from python_agent.cli import ToolUsageTracker, run_prompt


class ScriptedToolModel(BaseChatModel):
    responses: list[AIMessage]
    response_index: int = 0
    bound_tool_names: list[str] = []
    seen_messages: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        del tool_choice, kwargs
        self.bound_tool_names = [tool.name for tool in tools if isinstance(tool, BaseTool)]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, kwargs
        self.seen_messages.append(messages)
        response = self.responses[self.response_index]
        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_agent_executes_tool_calls_until_model_finishes(tmp_path: Path) -> None:
    model = ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write",
                        "args": {"path": "result.txt", "content": "created by the graph"},
                        "id": "write-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The file is ready."),
        ]
    )
    agent = create_agent(model, tmp_path)

    result = agent.invoke({"messages": [HumanMessage(content="Create result.txt")]})

    assert model.bound_tool_names == [
        "bash",
        "read",
        "search",
        "find_files",
        "write",
        "update",
        "delete",
    ]
    assert (tmp_path / "result.txt").read_text() == "created by the graph"
    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content == "The file is ready."


def test_run_prompt_returns_updated_history_and_text(tmp_path: Path) -> None:
    model = ScriptedToolModel(responses=[AIMessage(content="Hello from the agent.")])
    agent = create_agent(model, tmp_path)

    history, text = run_prompt(agent, [], "Hello")

    assert isinstance(history[0], HumanMessage)
    assert history[-1].content == "Hello from the agent."
    assert text == "Hello from the agent."


def test_run_prompt_reports_real_graph_tool_usage(tmp_path: Path) -> None:
    model = ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write",
                        "args": {"path": "tracked.txt", "content": "tracked"},
                        "id": "write-tracked",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Done."),
        ]
    )
    output: list[str] = []
    tracker = ToolUsageTracker(output=output.append)
    agent = create_agent(model, tmp_path)

    _, text = run_prompt(agent, [], "Create tracked.txt", tmp_path, tracker)

    assert text == "Done."
    assert tracker.tool_names == ["write"]
    assert output[0] == "[tool] write(path='tracked.txt', content=<7 chars>) started"
    assert any(line.startswith("[tool] write completed in ") for line in output)
    assert output[-1].startswith("[prompt] completed in ")


def test_run_prompt_attaches_tagged_content_without_retaining_it_in_history(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("flag = False\n", encoding="utf-8")
    model = ScriptedToolModel(responses=[AIMessage(content="Explained.")])
    agent = create_agent(model, tmp_path)

    history, _ = run_prompt(agent, [], "Explain @app.py", tmp_path)

    assert "flag = False" in str(model.seen_messages[0][1].content)
    assert history[0].content == "Explain @app.py"
