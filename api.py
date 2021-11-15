import os
import pathlib
from typing import Optional

import obsidiantools.api as otools
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import obsfeatures


class ObsidianPyLabRequest(BaseModel):
    vaultPath: str
    notePath: Optional[str] = None
    text: Optional[str] = None


# Setup app
app = FastAPI()

origins = ["app://obsidian.md"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available endpoints
available_fns = {"similar": lambda x: None}

# Read vault and notes into memory

VAULT_PATH = "/Users/taivo/kb"
vault = otools.Vault(pathlib.Path(VAULT_PATH)).connect()
notes = [
    obsfeatures.Note.from_path(name, VAULT_PATH / p)
    for name, p in vault.file_index.items()
]
print(f"{len(notes)} notes in vault")


@app.get("/")
def read_root():
    return {"scripts": [f"http://127.0.0.1:5000/function/{fn}" for fn in available_fns]}


@app.post("/similar")
def similar(request: ObsidianPyLabRequest):
    print(request)
    # items = [{"path": f"Commitment.md", "name": "Commitment", "info": {"score": 0.25}}]

    query_note_name = os.path.splitext(request.notePath)[0]
    query_note = [n for n in notes if n.name == query_note_name][0]

    print(query_note_name)
    print(query_note)
    result_df = (
        obsfeatures.jaccard_coefficients(query_note, vault.graph)
        .sort_values("jaccard", ascending=False)
        .head(20)
    )
    # result_df = obsfeatures.geodesic_distances(query_note, vault.graph)
    print(result_df.shape)

    items = []
    for _, row in result_df.iterrows():
        items.append(
            {
                "path": f'{row["name"]}.md',
                "name": row["name"],
                "info": {"score": row.jaccard},
            }
        )

    print(items)

    return {"contents": items}
