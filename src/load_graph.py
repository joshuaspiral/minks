"""TODO: DOCSTRING"""

import re
import os
from typing import Optional

from new_graph import KnowledgeGraph

def _parse(raw: str):
    """TODO: DOCSTRING"""
    # remove YAML front matter and embedded images
    content = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL)
    content = re.sub(r"!\[\[.*?\]\]", "", content)

    # make list of all links
    links = re.findall(r"\[\[(.+?)(?:\|.+?)?\]\]", content)
    links = [lk.strip() for lk in links]

    # remove link embeds, tags, markdown headers, formatting, and excessive whitespace
    content = re.sub(
        r"\[\[(.+?)(?:\|(.+?))?\]\]", lambda m: m.group(2) or m.group(1), content
    )
    content = re.sub(r"(?<!\w)#\w+", "", content)
    content = re.sub(r"#+\s*", "", content)
    content = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", content)
    content = re.sub(r"\s+", " ", content).strip()

    return content, links


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

        graph.add_note(name)

    for note in graph._notes.values():
        for

    return graph
