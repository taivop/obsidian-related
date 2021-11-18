import os
import pathlib

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
