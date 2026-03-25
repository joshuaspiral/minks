"""Knowledge graph implementation for representing an Obsidian note vault.

Defines classes for modelling notes and the links between them.
Each note is represented as a vertex, and links between notes are represented as edges.

Classes:
    _Note:
        Represents a single note containing a name, parsed content, and links to neighbouring notes.

    KnowledgeGraph:
        Stores a collection of notes and the links between them.
"""

from __future__ import annotations
from typing import Optional

# Graph implementation
class _Note:
    """A note in a knowledge graph, used to represent a note that would appear in an obsidian vault.

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

    def degree(self) -> int:
        """Return the degree of this note."""
        return len(self.links)

class KnowledgeGraph:
    """A graph used to represent an obsidian note vault.
    """
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
        if (name1 == name2):
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

    def get_all_note_names(self) -> set:
        """Return a set of all note names in this graph."""
        return set(self._notes.keys())
