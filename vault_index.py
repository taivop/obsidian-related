import os
import pathlib
import re

import networkx as nx
import obsidiantools.api as otools

from note import Note

re_aliased_wikilink = re.compile(r"\[\[(?!.+?:)[^\]\[]+\|([^\]\[]+)\]\]")


def _clean_wikilinks(s: str) -> str:
    """Replaces all wikilinks brackets in s."""
    # Remove aliased wikilinks
    chunks = re_aliased_wikilink.split(s)

    # Remove non-aliased wikilinks
    s2 = "".join(chunks).replace("[[", "").replace("]]", "")

    return s2


class VaultIndex:
    vault = None
    notes = None
    top2vec_model_path = pathlib.Path("data/top2vec_model")

    def __init__(self, vault_path: pathlib.Path, enable_top2vec=False):
        self.vault_path = vault_path
        self.enable_top2vec = enable_top2vec
        self.load()

    def __repr__(self):
        return f"VaultIndex at '{self.vault_path}', {len(self.notes)} notes"

    def load(self, reindex_top2vec=False):
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
        if self.enable_top2vec:
            # Import here so that we don't have to install top2vec if it's not used
            from top2vec import Top2Vec

            if reindex_top2vec or not os.path.exists(self.top2vec_model_path):
                self.index_top2vec()
            else:
                self.top2vec_model = Top2Vec.load(self.top2vec_model_path)

    def index_jaccard(self):
        """Compute Jaccard distances for all pairs."""
        self.jaccard_pairs = list(nx.jaccard_coefficient(self.graph_undirected))

    def index_top2vec(self):
        """Train top2vec word, document, and topic embeddings."""
        # Import here so that we don't have to install top2vec if it's not used
        from top2vec import Top2Vec

        notes = {
            name: note
            for (name, note) in self.notes.items()
            if len(note.plaintext) > 100
        }

        names = list(notes.keys())
        plaintexts = [_clean_wikilinks(note.plaintext) for note in notes.values()]

        hdbscan_args = {"min_cluster_size": 5}
        self.top2vec_model = Top2Vec(
            plaintexts,
            embedding_model="universal-sentence-encoder",
            use_embedding_model_tokenizer=True,
            keep_documents=False,
            document_ids=names,
            min_count=3,
            hdbscan_args=hdbscan_args,
        )
        self.top2vec_model.save(self.top2vec_model_path)
