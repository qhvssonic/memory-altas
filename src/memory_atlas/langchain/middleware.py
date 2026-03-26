"""MemoryAtlasMiddleware: LangChain 1.0 AgentMiddleware integration.

This is the primary user-facing interface. One line to add memory to any agent:

    agent = create_agent(
        model="openai:gpt-4o",
        tools=[...],
        middleware=[MemoryAtlasMiddleware(storage_path="./memory")],
    )
"""

from __future__ import annotations

from typing import Any

from memory_atlas.engine import MemoryEngine


class MemoryAtlasMiddleware:
    """LangChain 1.0 AgentMiddleware — game-engine style memory management.

    Hooks:
    - before_agent: Initialize session cache
    - before_model: Retrieve and inject relevant memories
    - after_model: Evaluate ingestion + scene management update
    - after_agent: Persist state + learn transition patterns
    """

    def __init__(self, storage_path: str = "./memory_atlas_data", **kwargs):
        self.engine = MemoryEngine(storage_path=storage_path, **kwargs)
        self._session_id: str = ""

    def before_agent(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        """Initialize session: load user profile and memory index into cache."""
        context = {}
        if runtime and hasattr(runtime, "context"):
            context = runtime.context
        self._session_id = context.get("session_id", "default")
        self.engine.scene.initialize_session(self._session_id)
        return None

    def before_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        """Retrieve relevant memories and inject into messages.

        Cache-first: hot → warm → cold (vector + tree search).
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # Get the latest user message
        last_msg = messages[-1]
        query = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
        if not query:
            return None

        # Scene manager retrieves optimal memory view
        memories = self.engine.scene.get_memory_view(query)
        if not memories:
            return None

        # Format and inject as system message
        context = self.engine.format_memories(memories)
        return {
            "messages": [{"role": "system", "content": context}],
        }

    def after_model(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        """Evaluate ingestion + run scene management cycle."""
        messages = state.get("messages", [])
        if not messages:
            return None

        # Extract recent turn content
        recent = self._extract_recent_turn(messages)
        if not recent:
            return None

        # Maybe ingest the recent turn
        self.engine.maybe_ingest(recent, self._session_id)

        # Scene management: prefetch + cull + LOD adjust
        # Extract entities from recent turn for scene update
        extraction = self.engine.extractor.extract(recent)
        entity_names = [e["name"] for e in extraction.entities]

        self.engine.scene.update(
            recent_message=recent,
            current_entities=entity_names,
        )
        return None

    def after_agent(self, state: dict, runtime: Any = None) -> dict[str, Any] | None:
        """Persist cache state and learn topic transition patterns."""
        self.engine.scene.persist()
        self.engine.scene.learn_transition_patterns()
        return None

    def _extract_recent_turn(self, messages: list) -> str:
        """Extract the most recent conversation turn."""
        if not messages:
            return ""
        # Take last 2 messages (user + assistant)
        recent = messages[-2:] if len(messages) >= 2 else messages
        parts = []
        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(msg))
        return "\n".join(parts)
