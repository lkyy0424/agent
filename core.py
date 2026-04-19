"""
agent/core.py — The main ReAct (Reasoning + Acting) agent loop.

Flow per iteration
------------------
1. Build the messages list from memory.
2. Call the LLM (via LLMClient).
3. If the response is a ToolCallResponse → execute tools, append results,
   loop again.
4. If the response is a TextResponse → we have a final answer; stop.
5. If MAX_ITERATIONS is reached → return a partial answer with a warning.

The core module is intentionally stateless between runs; all state lives in
AgentMemory, which is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

import config
from agent.executor import Executor
from agent.memory import AgentMemory
from agent.planner import Plan, Planner
from llm.client import LLMClient, TextResponse, ToolCallResponse
from tools.registry import ToolRegistry
from utils.logger import get_logger
from utils.parser import truncate

logger = get_logger(__name__)
console = Console()


@dataclass
class AgentResult:
    """Returned by AgentCore.run()."""

    answer: str
    iterations: int
    success: bool
    plan: Plan | None = None


class AgentCore:
    """
    Orchestrates the ReAct loop.

    Parameters
    ----------
    registry:   ToolRegistry with all tools pre-registered.
    memory:     AgentMemory for short-term + long-term state.
    use_planner: Whether to run the Planner before the main loop.
    verbose:    Whether to print Rich-formatted progress to stdout.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        memory: AgentMemory,
        use_planner: bool = True,
        verbose: bool = True,
    ) -> None:
        self._registry = registry
        self._memory = memory
        self._use_planner = use_planner
        self._verbose = verbose

        self._llm = LLMClient(tool_schemas=registry.schemas())
        self._executor = Executor(registry)
        self._planner = Planner(self._llm) if use_planner else None

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, task: str) -> AgentResult:
        """
        Execute *task* and return the agent's final answer.

        Parameters
        ----------
        task: Natural-language description of what the agent should do.
        """
        self._print_header(task)

        # ── Optional planning phase ───────────────────────────────────────────
        plan: Plan | None = None
        if self._use_planner and self._planner:
            plan = self._planner.create_plan(task)
            self._print_plan(plan)
            # Inject the plan into the conversation as context
            plan_context = (
                f"I have broken down your task into the following plan:\n\n"
                f"{plan.to_prompt_block()}\n\n"
                f"Now I will execute it step by step."
            )
            self._memory.short.add(self._llm.assistant_message(
                [{"type": "text", "text": plan_context}]
            ))

        # ── Main ReAct loop ───────────────────────────────────────────────────
        # Add the user's task as the first user message
        self._memory.short.add(self._llm.user_message(task))

        final_answer = ""
        iterations = 0

        for iteration in range(1, config.MAX_ITERATIONS + 1):
            iterations = iteration
            self._print_iteration(iteration)

            # ── LLM call ──────────────────────────────────────────────────────
            response = self._llm.chat(self._memory.short.all())

            if isinstance(response, TextResponse):
                # ── Final answer ───────────────────────────────────────────────
                final_answer = response.text
                self._print_final_answer(final_answer)
                # Save to memory so it's available in future sessions
                self._memory.long.set("last_answer", final_answer[:500])
                break

            if isinstance(response, ToolCallResponse):
                # ── Tool execution ─────────────────────────────────────────────
                # Append the assistant's tool-call message to history
                self._memory.short.add(
                    self._llm.assistant_message(response.raw_content)
                )

                # Execute all tool calls
                exec_results = self._executor.execute_all(response.tool_calls)

                # Print each tool call + result
                for er in exec_results:
                    self._print_tool_call(er.tool_name, er.input_args, er.output)

                # Build the tool-results message and add to history
                tool_results = [
                    {"tool_use_id": er.tool_use_id, "content": er.output}
                    for er in exec_results
                ]
                self._memory.short.add(
                    self._llm.tool_results_message(tool_results)
                )

        else:
            # Loop exhausted without a final answer
            final_answer = (
                f"Reached the maximum number of iterations ({config.MAX_ITERATIONS}) "
                "without producing a final answer. The last tool results are in the "
                "conversation history."
            )
            logger.warning("AgentCore: max iterations reached.")
            return AgentResult(
                answer=final_answer,
                iterations=iterations,
                success=False,
                plan=plan,
            )

        return AgentResult(
            answer=final_answer,
            iterations=iterations,
            success=True,
            plan=plan,
        )

    # ── Rich display helpers ──────────────────────────────────────────────────

    def _print_header(self, task: str) -> None:
        if not self._verbose:
            return
        console.print()
        console.print(
            Panel(
                f"[bold cyan]{task}[/bold cyan]",
                title="[bold]Action Agent[/bold]",
                border_style="cyan",
            )
        )

    def _print_plan(self, plan: Plan) -> None:
        if not self._verbose:
            return
        console.print(
            Panel(str(plan), title="[bold yellow]Plan[/bold yellow]", border_style="yellow")
        )

    def _print_iteration(self, n: int) -> None:
        if not self._verbose:
            return
        console.print(Rule(f"[dim]Iteration {n}[/dim]"))

    def _print_tool_call(
        self, name: str, args: dict[str, Any], result: str
    ) -> None:
        if not self._verbose:
            return
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        console.print(f"  [bold magenta]→ {name}[/bold magenta]({args_str})")
        console.print(
            Panel(
                truncate(result, 600),
                title="[dim]Tool result[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )

    def _print_final_answer(self, answer: str) -> None:
        if not self._verbose:
            return
        console.print()
        console.print(
            Panel(
                Markdown(answer),
                title="[bold green]Final Answer[/bold green]",
                border_style="green",
            )
        )
