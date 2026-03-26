"""Multi-level summarizer: generates L0 labels and L1 summaries."""

from __future__ import annotations

from dataclasses import dataclass

from memory_atlas.llm.provider import LLMProvider


@dataclass
class SummaryResult:
    """L0 label and L1 summary for a memory chunk."""

    label: str   # L0: ~20 tokens, one-line tag
    summary: str  # L1: ~80 tokens, concise summary


SUMMARY_PROMPT = """Generate a two-level summary for the following content.

Content:
---
{content}
---

Return a JSON object:
- "label": A single-line tag/label (~20 tokens max). Format: "YYYY-MM topic keyword summary"
- "summary": A concise summary (~80 tokens max) capturing key facts, decisions, and context.

Return ONLY valid JSON, no markdown."""


class Summarizer:
    """Generate L0 labels and L1 summaries for memory chunks."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def summarize(self, content: str, entities: list[str] | None = None) -> SummaryResult:
        """Generate L0 + L1 summaries for content."""
        if len(content.strip()) < 20:
            return SummaryResult(label=content.strip()[:80], summary=content.strip())

        try:
            data = self.llm.complete_json(
                SUMMARY_PROMPT.format(content=content[:3000]),
                system="You are a concise summarization assistant. Return only valid JSON.",
            )
            return SummaryResult(
                label=data.get("label", content[:80]),
                summary=data.get("summary", content[:200]),
            )
        except Exception:
            return self._rule_based_summary(content, entities)

    def _rule_based_summary(
        self, content: str, entities: list[str] | None = None
    ) -> SummaryResult:
        """Fallback rule-based summarization."""
        # L0: first meaningful line
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        first_line = lines[0] if lines else content[:80]
        label = first_line[:100]

        # L1: first few sentences
        sentences = content.replace("\n", " ").split(".")
        summary_parts = []
        token_count = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            est_tokens = len(s) // 4
            if token_count + est_tokens > 80:
                break
            summary_parts.append(s)
            token_count += est_tokens

        summary = ". ".join(summary_parts)
        if summary and not summary.endswith("."):
            summary += "."

        if entities:
            summary += f" Entities: {', '.join(entities[:5])}"

        return SummaryResult(label=label, summary=summary or label)
