"""LLM-based information extractor: entities, facts, decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory_atlas.llm.provider import LLMProvider


@dataclass
class ExtractionResult:
    """Extracted information from a chunk of content."""

    entities: list[dict[str, str]] = field(default_factory=list)  # [{name, type}]
    facts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    importance: float = 0.5


EXTRACTION_PROMPT = """Analyze the following content and extract structured information.

Content:
---
{content}
---

Return a JSON object with these fields:
- "entities": list of {{"name": "...", "type": "file|function|concept|person|project|tool"}}
- "facts": list of factual statements (strings)
- "decisions": list of decisions made (strings)
- "topics": list of topic keywords (strings)
- "importance": float 0-1 (how important is this for long-term memory)

Return ONLY valid JSON, no markdown."""


class Extractor:
    """Extract entities, facts, and decisions from content using LLM."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def extract(self, content: str) -> ExtractionResult:
        """Extract structured info from a content chunk."""
        if len(content.strip()) < 20:
            return ExtractionResult()

        try:
            data = self.llm.complete_json(
                EXTRACTION_PROMPT.format(content=content[:3000]),
                system="You are a precise information extraction assistant. Return only valid JSON.",
            )
            return ExtractionResult(
                entities=data.get("entities", []),
                facts=data.get("facts", []),
                decisions=data.get("decisions", []),
                topics=data.get("topics", []),
                importance=float(data.get("importance", 0.5)),
            )
        except Exception:
            return self._rule_based_extract(content)

    def _rule_based_extract(self, content: str) -> ExtractionResult:
        """Fallback rule-based extraction when LLM is unavailable."""
        import re

        # Simple entity extraction: capitalized words, file paths, function-like patterns
        entities: list[dict[str, str]] = []
        # File paths
        for m in re.finditer(r"[\w/]+\.\w{1,5}", content):
            entities.append({"name": m.group(), "type": "file"})
        # Function-like
        for m in re.finditer(r"\b(\w+)\s*\(", content):
            name = m.group(1)
            if len(name) > 2 and name[0].islower():
                entities.append({"name": name, "type": "function"})

        # Simple topic extraction from content
        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())
        word_freq: dict[str, int] = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        topics = sorted(word_freq, key=word_freq.get, reverse=True)[:5]

        importance = min(1.0, len(content) / 2000 * 0.5 + len(entities) * 0.1)

        return ExtractionResult(
            entities=entities[:10],
            topics=topics,
            importance=round(importance, 2),
        )
