"""Utilities for loading an Obsidian vault into a KnowledgeGraph.

Parses Markdown notes and constructs a graph representing the vault's structure.

Copyright (c) 2026 Caellum Yip Hoi-Lee, Catherine Abdul-Samad, Michael Chen, Joshua Yeung.
All rights reserved.
"""

import re
import os

from graph import KnowledgeGraph


def _parse(raw: str) -> tuple[str, list[str]]:
    """Return the cleaned text content and list of WikiLink targets found in raw.

    Applies the following transformations in order:

    1. Strips YAML front matter: removes the block delimited by --- lines at the
       start of the file (e.g. Obsidian metadata like tags, aliases, date).
       Pattern: ^---\\s*\\n.*?\\n---\\s*\\n with DOTALL so .* matches newlines.

    2. Removes embedded images: strips ![[...]] Obsidian image embeds so image
       filenames are not mistaken for note links or content.

    3. Extracts WikiLinks: finds all [[Target]] and [[Target|Alias]] patterns and
       records the target (left side of | if present) as a raw link string.

    4. Replaces WikiLinks with display text: substitutes [[Target|Alias]] with
       Alias and [[Target]] with Target, so the link syntax is removed but the
       human-readable label remains in the content.

    5. Removes hashtag tags: strips inline Obsidian tags like #topic that are not
       preceded by a word character, leaving ordinary English words containing #
       (e.g. C#) untouched.

    6. Removes Markdown headers: strips leading # characters and surrounding
       whitespace from header lines (e.g. ## Introduction becomes Introduction).

    7. Removes bold/italic markers: strips surrounding * and ** and *** from text
       while preserving the wrapped content.

    8. Collapses whitespace: replaces all runs of whitespace (spaces, newlines,
       tabs) with a single space and strips leading/trailing whitespace.

    Preconditions:
        - raw is a valid UTF-8 string
    """
    # 1. Strip YAML front matter (--- block at top of file)
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw, flags=re.DOTALL)

    # 2. Remove embedded image links e.g. ![[image.png]]
    content = re.sub(r"!\[\[.*?\]\]", "", content)

    # 3. Extract WikiLink targets e.g. [[Note]] or [[Note|Alias]] -> "Note"
    links = re.findall(r"\[\[(.+?)(?:\|.+?)?\]\]", content)
    links = [lk.strip() for lk in links]

    # 4. Replace WikiLinks with their display text
    content = re.sub(
        r"\[\[(.+?)(?:\|(.+?))?\]\]", lambda m: m.group(2) or m.group(1), content
    )

    # 5. Remove Obsidian hashtag tags e.g. #topic (but not C# or mid-word uses)
    content = re.sub(r"(?<!\w)#\w+", "", content)

    # 6. Remove Markdown header symbols e.g. ## Heading -> Heading
    content = re.sub(r"#+\s*", "", content)

    # 7. Remove bold/italic markers e.g. **word** -> word
    content = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", content)

    # 8. Collapse all whitespace runs into a single space
    content = re.sub(r"\s+", " ", content).strip()

    return content, links


def load_vault(vault_path: str) -> KnowledgeGraph:
    """Return a KnowledgeGraph built from all Markdown files in vault_path.

    Each .md file becomes a note vertex. WikiLinks between notes that resolve
    to an existing note in the vault become edges. Link resolution is
    case-insensitive and strips any folder path prefix (e.g. [[Folder/Note]]
    resolves to the note named Note).

    Files that are not .md are silently skipped. WikiLinks that point to notes
    not present in the vault are silently ignored.

    Preconditions:
        - vault_path is a valid path to a directory
    """
    graph = KnowledgeGraph()
    note_links: dict[str, list[str]] = {}

    for file_name in os.listdir(vault_path):
        if not file_name.endswith(".md"):
            continue

        with open(os.path.join(vault_path, file_name)) as file:
            raw = file.read()

        content, links = _parse(raw)
        name = file_name[:-3]
        graph.add_note(name, content)
        note_links[name] = links

    # Build a lowercase name -> original name lookup to handle case-insensitive WikiLinks.
    note_names = graph.get_all_note_names()
    lower_to_name = {n.lower(): n for n in note_names}

    for note, links in note_links.items():
        for neighbour in links:
            # Strip folder path prefix e.g. "Folder/Note" -> "Note"
            neighbour = neighbour.split("/")[-1].strip()
            resolved = lower_to_name.get(neighbour.lower())
            if resolved is not None and resolved != note:
                graph.add_link(note, resolved)

    return graph


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(
        config={
            "extra-imports": [
                "sentence_transformers",
                "plotly.graph_objects",
                "networkx",
                "plotly.subplots",
                "numpy",
                "sklearn.manifold",
                "graph",
                "similarity",
                "load_graph",
                "predictor",
                "visualize",
                "os",
                "re",
                "math",
                "random",
            ],
            "allowed-io": ["fit", "__init__", "main", "evaluate"],
            "max-line-length": 120,
        }
    )
