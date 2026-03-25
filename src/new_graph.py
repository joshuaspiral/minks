import re
import os
from dataclasses import dataclass, field
from typing import Optional
from __future__ import annotations

from assignments.project2.minks.src.graph import KnowledgeGraph

def _parse(raw: str):
    """TODO: DOCSTRING"""
    # remove YAML front matter and embedded images
    text = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL)
    text = re.sub(r"!\[\[.*?\]\]", "", text)

    # make list of all links
    links = re.findall(r"\[\[(.+?)(?:\|.+?)?\]\]", text)
    links = [lk.strip() for lk in links]

    # remove link embeds, tags, markdown headers, formatting, and excessive whitespace
    text = re.sub(
        r"\[\[(.+?)(?:\|(.+?))?\]\]", lambda m: m.group(2) or m.group(1), text
    )
    text = re.sub(r"(?<!\w)#\w+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return links, text


def load_vault(vault_path: str) -> KnowledgeGraph:
    """TODO: DOCSTRING"""
    graph = KnowledgeGraph()

    for file_name in os.listdir(vault_path):
        # skip anything that isn't a markdown file
        if not file_name.endswith('.md'):
            continue

        name = file_name[:-3]

        with open(os.path.join(vault_path, file_name)) as file:
            raw = file.read()

        # graph.add_note(name)

    # for loop, do graph.add_edge(name, other_name)

    return graph
