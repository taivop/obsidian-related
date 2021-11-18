import os
import re

import networkx as nx
import numpy as np
import pandas as pd

import vault_index
from note import Note

re_dailynote = re.compile(os.getenv("DAILY_NOTE_REGEX") or r"\d\d\d\d-\d\d-\d\d")


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
def geodesic_distances(note: Note, graph: nx.Graph) -> pd.DataFrame:
    """Get geodesic distances from note to all other notes."""
    all_shortest_paths = nx.single_source_shortest_path_length(graph, source=note.name)

    rows = []

    for name in graph.nodes:
        rows.append({"name": name, "distance": all_shortest_paths.get(name, np.inf)})

    return pd.DataFrame(rows).sort_values("distance")


def jaccard_coefficients(note: Note, index: vault_index.VaultIndex) -> pd.DataFrame:
    """Get jaccard coefficients from note to all other notes."""
    rows = []

    # TODO still wasteful because we go through all pairs at every query -- should move this logic to vault_index
    for n1, n2, jaccard in index.jaccard_pairs:
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
