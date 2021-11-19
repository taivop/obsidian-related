import pathlib

import networkx as nx
import obsidiantools.api as otools

from note import Note


class VaultIndex:
    vault = None
    notes = None

    def __init__(self, vault_path: pathlib.Path):
        self.vault_path = vault_path
        self.load()

    def __repr__(self):
        return f"VaultIndex at '{self.vault_path}', {len(self.notes)} notes"

    def load(self):
        """Read vault, metadata, notes and all indices into memory."""
        self.vault = otools.Vault(pathlib.Path(self.vault_path)).connect()

        # Convert graphs
        self.graph = self.vault.graph
        self.graph_undirected = nx.Graph(self.graph)

        # Notes to memory
        self.notes = {
            name: Note.from_path(name, self.vault_path / path)
            for name, path in self.vault.file_index.items()
        }

        # Build indices
        self.index_jaccard()

    def index_jaccard(self):
        """Compute Jaccard distances for all pairs."""
        self.jaccard_pairs = list(nx.jaccard_coefficient(self.graph_undirected))
