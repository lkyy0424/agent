"""
agent/memory.py — Dual-layer memory system for the agent.

Short-term memory
-----------------
Stores the full conversation history for the current session as a list of
Anthropic message dicts.  This is what gets sent to the API on every call.

Long-term memory
----------------
A simple JSON file on disk that persists key facts, summaries, or notes
across sessions.  The agent (or user code) can explicitly save and retrieve
entries.  Think of it as a lightweight key-value notebook.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Short-term memory (conversation history) ──────────────────────────────────

class ShortTermMemory:
    """
    Holds the in-session conversation history.

    The list it maintains is in Anthropic message format and can be passed
    directly to LLMClient.chat().
    """

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    def add(self, message: dict[str, Any]) -> None:
        """Append a message dict (role + content)."""
        self._history.append(message)

    def all(self) -> list[dict[str, Any]]:
        """Return the full conversation history."""
        return list(self._history)

    def clear(self) -> None:
        """Wipe the in-session history."""
        self._history.clear()
        logger.debug("Short-term memory cleared.")

    def last_n(self, n: int) -> list[dict[str, Any]]:
        """Return the last *n* messages."""
        return self._history[-n:]

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"ShortTermMemory({len(self._history)} messages)"


# ── Long-term memory (persistent JSON store) ──────────────────────────────────

class LongTermMemory:
    """
    Persistent key-value store backed by a JSON file.

    Each entry is a dict:
        {
            "key":       <str>,
            "value":     <any>,
            "timestamp": <ISO-8601 str>,
        }
    """

    def __init__(self, filename: str = "long_term.json") -> None:
        self._path: Path = config.MEMORY_DIR / filename
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                with self._path.open(encoding="utf-8") as f:
                    self._store = json.load(f)
                logger.debug("LongTermMemory: loaded %d entries from %s", len(self._store), self._path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LongTermMemory: failed to load '%s': %s", self._path, exc)
                self._store = {}
        else:
            self._store = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)
        logger.debug("LongTermMemory: saved %d entries to %s", len(self._store), self._path)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """Store or update a value under *key*."""
        self._store[key] = {
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()
        logger.info("LongTermMemory: set '%s'", key)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve the value for *key*, or *default* if not found."""
        entry = self._store.get(key)
        return entry["value"] if entry else default

    def delete(self, key: str) -> bool:
        """Remove *key* from the store.  Returns True if it existed."""
        if key in self._store:
            del self._store[key]
            self._save()
            logger.info("LongTermMemory: deleted '%s'", key)
            return True
        return False

    def list_keys(self) -> list[str]:
        return list(self._store.keys())

    def all_entries(self) -> list[dict[str, Any]]:
        return list(self._store.values())

    def clear(self) -> None:
        """Delete all entries (also removes the file)."""
        self._store = {}
        if self._path.exists():
            self._path.unlink()
        logger.info("LongTermMemory: cleared.")

    def summary(self) -> str:
        """Return a human-readable summary for the system prompt."""
        if not self._store:
            return "Long-term memory is empty."
        lines = ["Long-term memory entries:"]
        for entry in self._store.values():
            lines.append(f"  [{entry['timestamp'][:10]}] {entry['key']}: {entry['value']}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"LongTermMemory({len(self._store)} entries, path={self._path})"


# ── Combined memory facade ─────────────────────────────────────────────────────

class AgentMemory:
    """Convenience wrapper exposing both memory layers."""

    def __init__(self) -> None:
        self.short: ShortTermMemory = ShortTermMemory()
        self.long: LongTermMemory = LongTermMemory()

    def __repr__(self) -> str:
        return f"AgentMemory(short={self.short}, long={self.long})"
