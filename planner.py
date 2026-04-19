"""
agent/planner.py — Task planning and decomposition.

The Planner breaks a high-level user goal into an ordered list of sub-tasks
before the main agent loop starts executing.  The plan is informational
(shown to the user and stored in short-term memory) but does not rigidly
constrain the executor — the agent adapts if the situation changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubTask:
    """A single step in the plan."""

    index: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""

    def mark_done(self, result: str = "") -> None:
        self.status = TaskStatus.DONE
        self.result = result

    def mark_failed(self, reason: str = "") -> None:
        self.status = TaskStatus.FAILED
        self.result = reason

    def mark_in_progress(self) -> None:
        self.status = TaskStatus.IN_PROGRESS

    def __str__(self) -> str:
        icon = {
            TaskStatus.PENDING: "○",
            TaskStatus.IN_PROGRESS: "◑",
            TaskStatus.DONE: "●",
            TaskStatus.FAILED: "✗",
            TaskStatus.SKIPPED: "—",
        }[self.status]
        return f"{icon} [{self.index}] {self.description}"


@dataclass
class Plan:
    """An ordered collection of sub-tasks."""

    goal: str
    subtasks: list[SubTask] = field(default_factory=list)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_lines(cls, goal: str, lines: list[str]) -> "Plan":
        """Create a Plan from a list of step descriptions."""
        subtasks = [
            SubTask(index=i + 1, description=line.strip())
            for i, line in enumerate(lines)
            if line.strip()
        ]
        return cls(goal=goal, subtasks=subtasks)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def pending(self) -> list[SubTask]:
        return [t for t in self.subtasks if t.status == TaskStatus.PENDING]

    @property
    def done(self) -> list[SubTask]:
        return [t for t in self.subtasks if t.status == TaskStatus.DONE]

    @property
    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.DONE, TaskStatus.SKIPPED)
            for t in self.subtasks
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        header = f"Plan for: {self.goal}"
        steps = "\n".join(str(t) for t in self.subtasks)
        progress = f"{len(self.done)}/{len(self.subtasks)} steps complete"
        return f"{header}\n{steps}\n{progress}"

    def to_prompt_block(self) -> str:
        """Return a compact version suitable for injection into a prompt."""
        lines = [f"PLAN (goal: {self.goal}):"]
        for t in self.subtasks:
            lines.append(f"  Step {t.index}: {t.description}  [{t.status.value}]")
        return "\n".join(lines)


# ── Planner ───────────────────────────────────────────────────────────────────

class Planner:
    """
    Uses the LLM to decompose a high-level goal into concrete steps.

    The planner makes a single lightweight LLM call to generate the plan and
    then returns it.  The plan is NOT executed here.
    """

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    def create_plan(self, goal: str) -> Plan:
        """
        Ask the LLM to break *goal* into numbered steps.

        Returns
        -------
        A Plan object with PENDING sub-tasks.
        """
        logger.info("Planner: creating plan for goal=%r", goal[:80])

        planning_prompt = (
            f"You are a planning assistant. Break the following goal into "
            f"a numbered list of concrete, actionable steps. "
            f"Output ONLY the numbered list — no explanations, no headers.\n\n"
            f"Goal: {goal}"
        )

        messages = [{"role": "user", "content": planning_prompt}]
        response = self._llm.chat(messages)

        raw_text: str = getattr(response, "text", "")
        steps = self._parse_steps(raw_text)

        if not steps:
            # Fallback: single-step plan
            steps = [goal]

        plan = Plan.from_lines(goal, steps)
        logger.info("Planner: created %d-step plan", len(plan.subtasks))
        return plan

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_steps(text: str) -> list[str]:
        """
        Extract numbered or bulleted list items from the LLM response.
        Handles formats like: "1. Step", "1) Step", "• Step", "- Step"
        """
        pattern = re.compile(r"^\s*(?:\d+[.)]\s*|[-•*]\s+)(.+)$", re.MULTILINE)
        matches = pattern.findall(text)
        if matches:
            return [m.strip() for m in matches if m.strip()]

        # Fallback: each non-empty line is a step
        return [line.strip() for line in text.splitlines() if line.strip()]
