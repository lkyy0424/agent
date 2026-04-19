~"""
agent/executor.py — Action execution coordinator.

The Executor sits between the agent core loop and the ToolRegistry.
It receives a list of tool calls from the LLM, dispatches them (possibly
in parallel in the future), collects results, and formats them for the
next conversation turn.
"""

from __future__ import annotations

from typing import Any

from tools.registry import ToolRegistry
from utils.logger import get_logger
from utils.parser import truncate

logger = get_logger(__name__)


class ExecutionResult:
    """The result of executing one tool call."""

    def __init__(
        self,
        tool_use_id: str,
        tool_name: str,
        input_args: dict[str, Any],
        output: str,
    ) -> None:
        self.tool_use_id = tool_use_id
        self.tool_name = tool_name
        self.input_args = input_args
        self.output = output

    def __repr__(self) -> str:
        return (
            f"ExecutionResult(tool={self.tool_name!r}, "
            f"output={self.output[:60]!r}…)"
        )


class Executor:
    """
    Dispatches tool calls and collects results.

    Parameters
    ----------
    registry:   The ToolRegistry containing all registered tools.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute_all(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[ExecutionResult]:
        """
        Execute every tool call in *tool_calls* sequentially.

        Parameters
        ----------
        tool_calls: List of {"id": …, "name": …, "input": {…}} dicts,
                    as returned by LLMClient.

        Returns
        -------
        List of ExecutionResult objects in the same order.
        """
        results: list[ExecutionResult] = []
        for call in tool_calls:
            result = self._execute_one(call)
            results.append(result)
        return results

    def _execute_one(self, call: dict[str, Any]) -> ExecutionResult:
        tool_use_id: str = call["id"]
        tool_name: str = call["name"]
        input_args: dict[str, Any] = call.get("input", {})

        logger.info(
            "Executor: dispatching '%s' (id=%s) with args=%s",
            tool_name,
            tool_use_id,
            input_args,
        )

        output: str = self._registry.dispatch(tool_name, **input_args)

        logger.debug(
            "Executor: '%s' returned %d chars", tool_name, len(output)
        )

        return ExecutionResult(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            input_args=input_args,
            output=output,
        )

    def format_results_for_display(
        self, results: list[ExecutionResult], max_chars: int = 800
    ) -> str:
        """Return a human-readable summary of execution results."""
        lines: list[str] = []
        for r in results:
            lines.append(f"[Tool: {r.tool_name}]")
            lines.append(truncate(r.output, max_chars))
            lines.append("")
        return "\n".join(lines).strip()
