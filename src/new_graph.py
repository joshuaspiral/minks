"""TODO: DOCSTRING"""

from __future__ import annotations

# Graph implementation
class _Note:
    """TODO: DOCSTRING"""
    name: str
    content: str
    links: set[_Note]

    def __init__(self, name: str, raw: str):
        self.name = name
        self.content, self.links = _parse(raw)

    def degree(self) -> int:
        """Return the degree of this note."""
        return len(self.links)

    # def similarity_score():

class KnowledgeGraph:
    """TODO: DOCSTRING"""
    _notes: dict[str, _Note]

    def __init__(self) -> None:
        """Initialise an empty graph (no notes or links)."""
        self._notes = {}

    def add_note(self, name, raw):
        """Add a note with the given name and raw data.

        Do nothing if the given name is already in the graph.
        TODO: PRECONDITIONS
        """
        if name not in self._notes:
            self._notes[name] = _Note(name, raw)

    def add_link(self, name1: str, name2: str) -> None:
        """Add a link between two notes with the given names in this graph.

        Raise a ValueError if name1 or nam2 do not appear as notes in this graph.
        """
        if (name1 == name2):
            raise ValueError

        if name1 in self._notes and name2 in self._notes:
            note1 = self._notes[name1]
            note2 = self._notes[name2]

            note1.links.add(note2)
            note2.links.add(note1)
        else:
            raise ValueError

    def adjacent(self, name1: str, name2: str) -> bool:
        """Return whether name1 and name2 are adjacent notes in this graph.

        Return False if name1 or name2 do not appear as notes in this graph."""
        if name1 in self._notes and name2 in self._notes:
            note1 = self._notes[name1]
            return any(note2.name == name2 for note2 in note1.links)
        else:
            return False

    def get_neighbours(self, name: str) -> set:
        """Return a set of the neighbours of the given name.

        Note that the *names* are returned, not the _Note objects themselves.

        Raise a ValueError if name does not appear as a vertex in this graph.
        """
        if name in self._notes:
            note = self._notes[name]
            return {neighbour.name for neighbour in note.links}
        else:
            raise ValueError

    def get_all_note_names(self) -> set:
        """Return a set of all note names in this graph."""
        return set(self._notes.keys())
