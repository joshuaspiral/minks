"""Utilities for loading an Obsidian vault into a KnowledgeGraph.

This module provides functionality for parsing Markdown notes from an Obsidian vault and
constructing a KnowledgeGraph representing the vault's structure.
Each note becomes a vertex in the graph, and wiki-style links (e.g., [[Note Name]]) between notes become edges.

Functions:
    _parse:
        Cleans raw Markdown note content and extracts internal note links.

    load_vault:
        Reads all Markdown files in a vault directory and builds a KnowledgeGraph containing the notes and their links.
"""

import re
import os

from graph import KnowledgeGraph


def _parse(raw: str):
    """Parse the raw data into something we can work with to analyse content.

    Remove YAML front matter, embedded images, link embeds, tags, markdown headers,
    formatting, and excessive whitespace.

    Return the cleaned content as well as a list of all the links the note made to other notes.
    """
    content = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL)
    content = re.sub(r"!\[\[.*?\]\]", "", content)

    links = re.findall(r"\[\[(.+?)(?:\|.+?)?\]\]", content)
    links = [lk.strip() for lk in links]

    content = re.sub(
        r"\[\[(.+?)(?:\|(.+?))?\]\]", lambda m: m.group(2) or m.group(1), content
    )
    content = re.sub(r"(?<!\w)#\w+", "", content)
    content = re.sub(r"#+\s*", "", content)
    content = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", content)
    content = re.sub(r"\s+", " ", content).strip()

    return content, links


def load_vault(vault_path: str) -> KnowledgeGraph:
    """Load all the notes from a vault onto a KnowledgeGraph and returns it.

    Preconditions:
        - vault_path is a valid file path to a vault
    """
    graph = KnowledgeGraph()
    note_links: dict[str, list[str]] = {}

    # Load all notes
    for file_name in os.listdir(vault_path):
        if not file_name.endswith(".md"):
            continue

        with open(os.path.join(vault_path, file_name)) as file:
            raw = file.read()

        content, links = _parse(raw)
        name = file_name[:-3]
        graph.add_note(name, content)
        note_links[name] = links

    # Add all links
    note_names = graph.get_all_note_names()
    for note, links in note_links.items():
        for neighbour in links:
            if neighbour in note_names:
                graph.add_link(note, neighbour)

    return graph
