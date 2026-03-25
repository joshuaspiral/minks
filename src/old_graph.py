import re
import math
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Note:
    name: str
    raw: str
    content: str = ""
    links: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.links, self.content = _parse(self.raw)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Note) and self.name == other.name


def _parse(raw: str) -> tuple[list[str], str]:
    text = raw
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    links = re.findall(r"\[\[(.+?)(?:\|.+?)?\]\]", text)
    links = [lk.strip() for lk in links]
    text = re.sub(
        r"\[\[(.+?)(?:\|(.+?))?\]\]", lambda m: m.group(2) or m.group(1), text
    )
    text = re.sub(r"(?<!\w)#\w+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return links, text


def load_vault(vault_path: str) -> "KnowledgeGraph":
    g = KnowledgeGraph()
    for fname in os.listdir(vault_path):
        if not fname.endswith(".md"):
            continue
        name = fname[:-3]
        with open(os.path.join(vault_path, fname)) as f:
            raw = f.read()
        g._notes[name] = Note(name=name, raw=raw)
        g._adj[name] = set()
    for note in g._notes.values():
        for target in note.links:
            if target in g._notes:
                g._adj[note.name].add(target)
                g._adj[target].add(note.name)
    return g


class KnowledgeGraph:
    def __init__(self):
        self._notes: dict[str, Note] = {}
        self._adj: dict[str, set[str]] = {}

    def add_predicted_edges(self, predictions: list[tuple[str, str, float]]):
        for u, v, _ in predictions:
            self._adj[u].add(v)
            self._adj[v].add(u)

    @property
    def nodes(self) -> list[str]:
        return list(self._notes.keys())

    @property
    def edges(self) -> list[tuple[str, str]]:
        seen = set()
        result = []
        for u, nbrs in self._adj.items():
            for v in nbrs:
                key = tuple(sorted([u, v]))
                if key not in seen:
                    seen.add(key)
                    result.append((u, v))
        return result

    def neighbours(self, node: str) -> set[str]:
        return self._adj.get(node, set())

    def has_edge(self, u: str, v: str) -> bool:
        return v in self._adj.get(u, set())

    def degree(self, node: str) -> int:
        return len(self._adj.get(node, set()))

    def get_note(self, name: str) -> Optional[Note]:
        return self._notes.get(name)

    def non_edges(self) -> list[tuple[str, str]]:
        nodes = self.nodes
        result = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not self.has_edge(u, v):
                    result.append((u, v))
        return result

    def degree_centrality(self) -> dict[str, float]:
        n = len(self._notes)
        if n <= 1:
            return {node: 0.0 for node in self._notes}
        return {node: self.degree(node) / (n - 1) for node in self._notes}

    def connected_components(self) -> list[set[str]]:
        # iterative dfs
        visited = set()
        components = []
        for start in self._notes:
            if start in visited:
                continue
            comp = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                stack.extend(self._adj[node] - visited)
            components.append(comp)
        return components

    def copy(self) -> "KnowledgeGraph":
        g = KnowledgeGraph()
        for name, note in self._notes.items():
            g._notes[name] = note
        for name, nbrs in self._adj.items():
            g._adj[name] = set(nbrs)
        return g

    def remove_edge(self, u: str, v: str):
        self._adj[u].discard(v)
        self._adj[v].discard(u)

    def __repr__(self):
        return f"KnowledgeGraph(nodes={len(self._notes)}, edges={len(self.edges)})"
