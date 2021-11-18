import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.shortest_paths.generic import all_shortest_paths
from obsidiantools.md_utils import _get_ascii_plaintext_from_md_file

import vault_index

re_dailynote = re.compile(os.getenv("DAILY_NOTE_REGEX") or r"\d\d\d\d-\d\d-\d\d")


@dataclass
class Note:
    name: str
    path: Path
    md: str
    plaintext: str

    @staticmethod
    def from_path(name: str, path: Path) -> "Note":
        with path.open() as f:
            md = f.read()
            plaintext = _get_ascii_plaintext_from_md_file(path)
        return Note(name, path, md, plaintext)


def get_notes_individual_df(index: vault_index.VaultIndex) -> pd.DataFrame:
    """Get individual features for a list of notes."""
    return pd.DataFrame(
        [get_note_individual_features(n, index.vault) for n in index.notes.values()]
    )


def get_note_individual_features(note: Note, vault) -> dict:
    """Get features for a single note."""
    return {
        "name": note.name,
        "name_n_char": name_n_char(note),
        "name_n_words": name_n_words(note),
        "plaintext_n_char": plaintext_n_char(note),
        "plaintext_n_words": plaintext_n_words(note),
        "is_daily": is_daily(note),
        "exists": note.name not in vault.nonexistent_notes,
    }


# === Function of note only ===
def is_daily(note: Note) -> bool:
    """True if note looks like daily note."""
    return re_dailynote.match(note.name) is not None


def name_n_char(note: Note) -> int:
    """Number of characters in note name."""
    return len(note.name)


def name_n_words(note: Note) -> int:
    """Number of words in note name."""
    return len(note.name.split())


def plaintext_n_char(note: Note) -> int:
    """Number of characters in note plaintext content."""
    return len(note.plaintext)


def plaintext_n_words(note: Note) -> int:
    """Number of words in note plaintext content."""
    return len(note.plaintext.split())


# === Function of graph and note ===
def geodesic_distances(note: Note, graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Get geodesic distances from note to all other notes."""
    g = nx.Graph(graph)  # convert to undirected graph
    all_shortest_paths = nx.single_source_shortest_path_length(g, source=note.name)

    rows = []

    for name in g.nodes:
        rows.append({"name": name, "distance": all_shortest_paths.get(name, np.inf)})

    return pd.DataFrame(rows).sort_values("distance")


def jaccard_coefficients(note: Note, graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Get jaccard coefficients from note to all other notes."""
    g = nx.Graph(graph)  # convert to undirected graph
    # TODO this is wasteful because the jaccard coefficients don't have to be recalculated for every query
    jaccard_coefficients = nx.jaccard_coefficient(g)

    rows = []

    for n1, n2, jaccard in jaccard_coefficients:
        if n1 == note.name:
            rows.append({"name": n2, "jaccard": jaccard})
        elif n2 == note.name:
            rows.append({"name": n1, "jaccard": jaccard})

    return pd.DataFrame(rows)


# === Function of graph, but not note ===
def pagerank_centralities(graph: nx.Graph) -> pd.DataFrame:
    """Get pagerank centralities for all notes."""
    pr = nx.pagerank(graph)

    rows = []

    for name in graph.nodes:
        rows.append({"name": name, "pagerank": pr[name]})

    return pd.DataFrame(rows)
