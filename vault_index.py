import os
import pathlib

import networkx as nx
import obsidiantools.api as otools

import obsfeatures


def load_vault(vault_path: pathlib.Path):
    vault = otools.Vault(pathlib.Path(vault_path)).connect()
    notes = [
        obsfeatures.Note.from_path(name, vault_path / p)
        for name, p in vault.file_index.items()
    ]
    print(f"{len(notes)} notes in vault")

    return vault, notes


class VaultIndex:
    vault = None
    notes = None

    def __init__(self, vault_path: pathlib.Path):
        self.vault_path = vault_path
        self.load()
        self.index_jaccard()

    def load(self):
        """Read vault, metadata, notes and all indices into memory."""
        self.vault = otools.Vault(pathlib.Path(self.vault_path)).connect()
        self.notes = {
            name: obsfeatures.Note.from_path(name, self.vault_path / path)
            for name, path in self.vault.file_index.items()
        }
        self.index_jaccard()

    def index_jaccard(self):
        """Compute Jaccard distances for all pairs."""
        self.jaccard_coefficients = nx.jaccard_coefficient(nx.Graph(self.vault.graph))