"""Tree-based semantic index for memory navigation (PageIndex-style)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TreeNode:
    """A node in the semantic tree index."""

    id: str
    label: str
    summary: str = ""
    node_type: str = "topic"  # root / category / topic / memory
    depth: int = 0
    children: list[TreeNode] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "summary": self.summary,
            "node_type": self.node_type,
            "depth": self.depth,
            "memory_ids": self.memory_ids,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeNode:
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            id=data["id"],
            label=data["label"],
            summary=data.get("summary", ""),
            node_type=data.get("node_type", "topic"),
            depth=data.get("depth", 0),
            children=children,
            memory_ids=data.get("memory_ids", []),
        )


class TreeIndex:
    """JSON-based tree index for hierarchical memory navigation."""

    def __init__(self, storage_path: str | Path):
        self.file_path = Path(storage_path) / "tree_index.json"
        self.root = self._load()

    def _load(self) -> TreeNode:
        if self.file_path.exists():
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            return TreeNode.from_dict(data)
        return TreeNode(id="root", label="Memory Root", node_type="root", depth=0)

    def save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps(self.root.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def find_node(self, node_id: str, start: TreeNode | None = None) -> TreeNode | None:
        """DFS search for a node by ID."""
        node = start or self.root
        if node.id == node_id:
            return node
        for child in node.children:
            found = self.find_node(node_id, child)
            if found:
                return found
        return None

    def add_child(self, parent_id: str, child: TreeNode) -> bool:
        """Add a child node under the given parent."""
        parent = self.find_node(parent_id)
        if not parent:
            return False
        child.depth = parent.depth + 1
        parent.children.append(child)
        self.save()
        return True

    def add_memory_to_node(self, node_id: str, memory_id: str) -> bool:
        """Associate a memory ID with a tree node."""
        node = self.find_node(node_id)
        if not node:
            return False
        if memory_id not in node.memory_ids:
            node.memory_ids.append(memory_id)
            self.save()
        return True

    def get_node_path(self, node_id: str) -> list[TreeNode]:
        """Get the path from root to the given node."""
        path: list[TreeNode] = []
        self._find_path(self.root, node_id, path)
        return path

    def _find_path(
        self, node: TreeNode, target_id: str, path: list[TreeNode]
    ) -> bool:
        path.append(node)
        if node.id == target_id:
            return True
        for child in node.children:
            if self._find_path(child, target_id, path):
                return True
        path.pop()
        return False

    def get_outline(self, max_depth: int = 2) -> str:
        """Return a text outline of the tree for LLM navigation."""
        lines: list[str] = []
        self._outline(self.root, lines, max_depth)
        return "\n".join(lines)

    def _outline(
        self, node: TreeNode, lines: list[str], max_depth: int
    ) -> None:
        indent = "  " * node.depth
        mem_count = len(node.memory_ids)
        suffix = f" ({mem_count} memories)" if mem_count else ""
        lines.append(f"{indent}- [{node.id}] {node.label}{suffix}")
        if node.depth < max_depth:
            for child in node.children:
                self._outline(child, lines, max_depth)
