"""Tree-based reasoning search: LLM navigates the semantic tree index."""

from __future__ import annotations

from memory_atlas.core.registry import Registry
from memory_atlas.core.tree_index import TreeIndex
from memory_atlas.llm.provider import LLMProvider
from memory_atlas.retrieval.vector_search import SearchResult


NAV_PROMPT = """You are navigating a memory tree index to find relevant memories.

Query: {query}

Current node children:
{children}

Which children are most likely to contain memories relevant to the query?
Return a JSON list of node IDs to explore, ordered by relevance. Max {max_branches} IDs.
Return ONLY a JSON array of strings, e.g. ["node1", "node2"]"""


class TreeSearch:
    """LLM-guided navigation of the tree index to find relevant memories."""

    def __init__(self, registry: Registry, tree: TreeIndex, llm: LLMProvider):
        self.registry = registry
        self.tree = tree
        self.llm = llm

    def search(
        self, query: str, top_k: int = 10, max_depth: int = 3, max_branches: int = 3
    ) -> list[SearchResult]:
        """Navigate tree from root, using LLM to choose branches."""
        results: list[SearchResult] = []
        visited: set[str] = set()
        self._navigate(
            query, self.tree.root, results, visited,
            top_k, max_depth, max_branches, current_depth=0,
        )
        return results[:top_k]

    def _navigate(
        self,
        query: str,
        node,
        results: list[SearchResult],
        visited: set[str],
        top_k: int,
        max_depth: int,
        max_branches: int,
        current_depth: int,
    ) -> None:
        if node.id in visited or current_depth > max_depth or len(results) >= top_k:
            return
        visited.add(node.id)

        # Collect memories at this node
        for mid in node.memory_ids:
            rec = self.registry.get_memory(mid)
            if rec:
                # Score based on depth (deeper = more specific = higher score)
                score = 0.5 + (current_depth * 0.1)
                results.append(SearchResult(record=rec, score=score, source="tree"))

        # Navigate children
        if not node.children:
            return

        children_desc = "\n".join(
            f"- [{c.id}] {c.label}: {c.summary or '(no summary)'} "
            f"({len(c.memory_ids)} memories, {len(c.children)} sub-topics)"
            for c in node.children
        )

        try:
            chosen_ids = self.llm.complete_json(
                NAV_PROMPT.format(
                    query=query, children=children_desc, max_branches=max_branches
                ),
                system="Return only a JSON array of node ID strings.",
            )
            if isinstance(chosen_ids, list):
                for cid in chosen_ids[:max_branches]:
                    child = next((c for c in node.children if c.id == cid), None)
                    if child:
                        self._navigate(
                            query, child, results, visited,
                            top_k, max_depth, max_branches, current_depth + 1,
                        )
        except Exception:
            # Fallback: visit all children
            for child in node.children[:max_branches]:
                self._navigate(
                    query, child, results, visited,
                    top_k, max_depth, max_branches, current_depth + 1,
                )
