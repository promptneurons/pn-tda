"""Intra-document graph structure from markdown heading hierarchy.

Parses markdown headings into a tree, then produces edges representing
the document's internal topology. Heading level encodes filtration depth:
- h1 appears at filtration 0 (broadest)
- h2 at 1/max_depth, h3 at 2/max_depth, etc.

Nodes are (doc_id, heading_text) pairs. Edges connect:
- Parent → child headings (h2 under h1)
- Sibling headings at the same level under the same parent

This captures the author's mental model of a topic's structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#{1,6})?\s*$', re.MULTILINE)


@dataclass
class HeadingNode:
    """A heading in the document outline tree."""
    level: int          # 1-6
    text: str           # heading text
    children: list["HeadingNode"] = field(default_factory=list)
    parent: "HeadingNode | None" = field(default=None, repr=False)


def parse_headings(markdown: str) -> list[HeadingNode]:
    """Parse markdown into a flat list of HeadingNode."""
    nodes = []
    for match in HEADING_RE.finditer(markdown):
        level = len(match.group(1))
        text = match.group(2).strip()
        nodes.append(HeadingNode(level=level, text=text))
    return nodes


def build_heading_tree(headings: list[HeadingNode]) -> list[HeadingNode]:
    """Build a tree from flat heading list using level nesting.

    Returns the list of root-level headings. Each node's .children
    and .parent fields are populated.
    """
    if not headings:
        return []

    roots: list[HeadingNode] = []
    stack: list[HeadingNode] = []

    for node in headings:
        # Pop stack until we find a parent with lower level
        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        else:
            roots.append(node)

        stack.append(node)

    return roots


@dataclass
class HeadingEdge:
    """An edge in the heading graph."""
    source: str  # node label
    target: str  # node label
    edge_type: str  # "parent_child" or "sibling"
    filtration: float  # normalized depth [0, 1]


def extract_heading_edges(
    doc_id: str,
    markdown: str,
) -> tuple[list[str], list[HeadingEdge]]:
    """Extract intra-document edges from heading hierarchy.

    Returns (node_labels, edges) where node labels are "{doc_id}#{heading_text}"
    and edges encode parent-child and sibling relationships.

    Filtration value = heading_level / max_level, so deeper headings
    appear later in the filtration (more specific = higher filtration).
    """
    headings = parse_headings(markdown)
    if not headings:
        return [doc_id], []

    roots = build_heading_tree(headings)
    max_level = max(h.level for h in headings)

    nodes: list[str] = [doc_id]  # document root
    edges: list[HeadingEdge] = []
    seen_labels: dict[str, int] = {}

    def node_label(heading: HeadingNode) -> str:
        base = f"{doc_id}#{heading.text}"
        if base in seen_labels:
            seen_labels[base] += 1
            return f"{base} ({seen_labels[base]})"
        seen_labels[base] = 1
        return base

    def walk(node: HeadingNode, parent_label: str):
        label = node_label(node)
        nodes.append(label)
        filt = node.level / max_level if max_level > 0 else 0.0

        # Parent → child edge
        edges.append(HeadingEdge(
            source=parent_label,
            target=label,
            edge_type="parent_child",
            filtration=filt,
        ))

        # Sibling edges (children at same level under same parent)
        for i, child in enumerate(node.children):
            walk(child, label)
            # Connect consecutive siblings
            if i > 0:
                prev_label = node_label_cache[id(node.children[i - 1])]
                curr_label = node_label_cache[id(child)]
                edges.append(HeadingEdge(
                    source=prev_label,
                    target=curr_label,
                    edge_type="sibling",
                    filtration=child.level / max_level if max_level > 0 else 0.0,
                ))

    # Cache node labels for sibling linking
    node_label_cache: dict[int, str] = {}

    def walk_with_cache(node: HeadingNode, parent_label: str):
        label = node_label(node)
        node_label_cache[id(node)] = label
        nodes.append(label)
        filt = node.level / max_level if max_level > 0 else 0.0

        edges.append(HeadingEdge(
            source=parent_label,
            target=label,
            edge_type="parent_child",
            filtration=filt,
        ))

        for i, child in enumerate(node.children):
            walk_with_cache(child, label)
            if i > 0:
                prev = node_label_cache[id(node.children[i - 1])]
                curr = node_label_cache[id(child)]
                edges.append(HeadingEdge(
                    source=prev,
                    target=curr,
                    edge_type="sibling",
                    filtration=child.level / max_level if max_level > 0 else 0.0,
                ))

    # Reset and use the cached version
    nodes = [doc_id]
    edges = []
    seen_labels = {}
    for root in roots:
        walk_with_cache(root, doc_id)

    return nodes, edges


def heading_depth_stats(markdown: str) -> dict:
    """Compute heading hierarchy statistics for a document.

    Returns:
        {
            "heading_count": int,
            "max_depth": int,
            "depth_distribution": dict[int, int],  # level → count
            "branching_factor": float,  # avg children per non-leaf
            "leaf_fraction": float,  # fraction of headings with no children
        }
    """
    headings = parse_headings(markdown)
    if not headings:
        return {
            "heading_count": 0, "max_depth": 0,
            "depth_distribution": {}, "branching_factor": 0.0,
            "leaf_fraction": 1.0,
        }

    roots = build_heading_tree(headings)

    depth_dist: dict[int, int] = {}
    for h in headings:
        depth_dist[h.level] = depth_dist.get(h.level, 0) + 1

    non_leaves = [h for h in headings if h.children]
    leaves = [h for h in headings if not h.children]

    branching = (
        sum(len(h.children) for h in non_leaves) / len(non_leaves)
        if non_leaves else 0.0
    )

    return {
        "heading_count": len(headings),
        "max_depth": max(h.level for h in headings),
        "depth_distribution": depth_dist,
        "branching_factor": branching,
        "leaf_fraction": len(leaves) / len(headings) if headings else 1.0,
    }
