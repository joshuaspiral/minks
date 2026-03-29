"""Knowledge graph implementation for representing an Obsidian note vault.

Defines classes for modelling notes and the links between them.

Copyright (c) 2026 Caellum Yip Hoi-Lee, Catherine Abdul-Samad, Michael Chen, Joshua Yeung.
All rights reserved.
"""

from __future__ import annotations


class _Note:
    """A note in a knowledge graph, used to represent a note that would appear in an Obsidian vault.

    Instance Attributes:
        - name: The name of the file containing the note.
        - content: The cleaned/parsed version of the content contained in the original note.
        - links: The notes that are adjacent/linked to this note.

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.links for u in self.links)
    """

    name: str
    content: str
    links: set[_Note]

    def __init__(self, name: str, content: str):
        """Initialise a new note with the given name and content.

        This note is initialised with no neighbours/links.
        """

        self.name = name
        self.content = content
        self.links = set()

    def get_degree(self) -> int:
        """Return the degree of this note."""
        return len(self.links)

    def get_connected_component(self, visited: set[str]) -> set[str]:
        """Return the set of all note names reachable from this note using recursive DFS.

        visited is the set of note names that have already been visited.
        """
        visited.add(self.name)
        for neighbour in self.links:
            if neighbour.name not in visited:
                neighbour.get_connected_component(visited)
        return visited


class KnowledgeGraph:
    """A graph used to represent an obsidian note vault."""

    # Private Instance Attributes:
    #     - _notes:
    #         A collection of the notes contained in this graph.
    #         Maps name to _Note object.
    _notes: dict[str, _Note]

    def __init__(self) -> None:
        """Initialise an empty graph (no notes or links)."""
        self._notes = {}

    def add_note(self, name: str, content: str):
        """Add a note with the given name and raw data.

        Do nothing if the given name is already in the graph.
        """
        if name not in self._notes:
            self._notes[name] = _Note(name, content)

    def add_link(self, name1: str, name2: str) -> None:
        """Add a link between two notes with the given names in this graph.

        Raise a ValueError if name1 or nam2 do not appear as notes in this graph.
        """
        if name1 == name2:
            raise ValueError(f"Self-links are not permitted: '{name1}'")

        if name1 not in self._notes or name2 not in self._notes:
            raise ValueError(f"One or both notes not found: '{name1}', '{name2}'")

        note1 = self._notes[name1]
        note2 = self._notes[name2]
        note1.links.add(note2)
        note2.links.add(note1)

    def adjacent(self, name1: str, name2: str) -> bool:
        """Return whether name1 and name2 are adjacent notes in this graph.

        Return False if name1 or name2 do not appear as notes in this graph."""
        if name1 in self._notes and name2 in self._notes:
            note1 = self._notes[name1]
            return any(note2.name == name2 for note2 in note1.links)

        return False

    def get_neighbours(self, name: str) -> set:
        """Return a set of the neighbours of the given name.

        Note that the *names* are returned, not the _Note objects themselves.

        Raise a ValueError if name does not appear as a vertex in this graph.
        """
        if name in self._notes:
            note = self._notes[name]
            return {neighbour.name for neighbour in note.links}

        raise ValueError(f"Note not found: '{name}'")

    def get_note(self, name: str) -> _Note:
        """
        Returns the _Note object with the given name.
        """
        return self._notes[name]

    def __str__(self) -> str:
        """Return a string representation showing note and edge counts."""
        return f"KnowledgeGraph({len(self._notes)} notes, {len(self.get_edges())} edges)"

    def get_all_note_names(self) -> set:
        """Return a set of all note names in this graph."""
        return set(self._notes.keys())

    def get_notes(self) -> list[str]:
        """Return a list of all note names in this graph."""
        return list(self._notes.keys())

    def get_edges(self) -> list[tuple[str, str]]:
        """Return a list of all edges as (name1, name2) tuples, each pair appearing once."""
        seen = set()
        result = []
        for name, note in self._notes.items():
            for neighbour in note.links:
                edge = tuple(sorted((name, neighbour.name)))
                if edge not in seen:
                    seen.add(edge)
                    result.append(edge)
        return result

    def degree(self, name: str) -> int:
        """Return the degree of the note with the given name.

        Raise a ValueError if name does not appear as a note in this graph.
        """
        if name not in self._notes:
            raise ValueError(f"Note not found: '{name}'")
        return self._notes[name].get_degree()

    def non_edges(self) -> list[tuple[str, str]]:
        """Return a list of all non-existent edges (pairs of unlinked, distinct notes)."""
        names = self.get_notes()
        existing = set(self.get_edges())
        result = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                edge = tuple(sorted((names[i], names[j])))
                if edge not in existing:
                    result.append((names[i], names[j]))
        return result

    def remove_edge(self, name1: str, name2: str) -> None:
        """Remove the edge between name1 and name2.

        Raise a ValueError if either note does not exist or they are not linked.
        """
        if name1 not in self._notes or name2 not in self._notes:
            raise ValueError(f"One or both notes not found: '{name1}', '{name2}'")
        note1 = self._notes[name1]
        note2 = self._notes[name2]
        note1.links.discard(note2)
        note2.links.discard(note1)

    def copy(self) -> KnowledgeGraph:
        """Return a deep copy of this graph."""
        new_graph = KnowledgeGraph()
        for name, note in self._notes.items():
            new_graph.add_note(name, note.content)
        for name1, name2 in self.get_edges():
            new_graph.add_link(name1, name2)
        return new_graph

    def add_predicted_edges(self, predictions: list[tuple[str, str, float]]) -> None:
        """Add edges from a list of (name1, name2, score) prediction tuples."""
        for name1, name2, _ in predictions:
            if not self.adjacent(name1, name2):
                self.add_link(name1, name2)

    def degree_centrality(self) -> dict[str, float]:
        """Return a dictionary mapping note names to their degree centrality scores."""
        n = len(self._notes)
        if n <= 1:
            return {name: 0.0 for name in self._notes}
        return {name: note.get_degree() / (n - 1) for name, note in self._notes.items()}

    def connected_components(self) -> list[set[str]]:
        """Return a list of sets, each containing the note names in one connected component."""
        visited = set()
        components = []
        for name, note in self._notes.items():
            if name not in visited:
                component = note.get_connected_component(visited)
                components.append(component.copy())
        return components
