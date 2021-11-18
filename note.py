from dataclasses import dataclass
from pathlib import Path

from obsidiantools.md_utils import _get_ascii_plaintext_from_md_file


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
