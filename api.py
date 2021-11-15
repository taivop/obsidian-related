import os
import pathlib
from typing import Optional

import numpy as np
import obsidiantools.api as otools
import pandas as pd
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
available_fns = ["similar"]

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


def _df_to_items(df: pd.DataFrame, score_getter):
    items = []

    for _, row in df.iterrows():
        items.append(
            {
                "path": f'{row["name"]}.md',
                "name": row["name"],
                "info": {"score": score_getter(row)},
            }
        )

    return items


def get_items_jaccard(query_note, n_items=10):
    result_df = (
        obsfeatures.jaccard_coefficients(query_note, vault.graph)
        .sort_values("jaccard", ascending=False)
        .head(n_items)
    )

    return _df_to_items(result_df, lambda row: row.jaccard)


def _features_merged(query_note, graph):
    jaccard_df = obsfeatures.jaccard_coefficients(query_note, graph)
    note_individual_features = obsfeatures.get_notes_individual_df(notes)
    df = jaccard_df.merge(note_individual_features, on="name")

    return df


def get_items_jaccard_short(feature_df, n_items=5):
    result_df = feature_df
    print(result_df["is_daily"].dtype)
    result_df = result_df[result_df["is_daily"] == False]
    print(result_df.head(5))
    result_df = result_df[result_df["name_n_words"] <= 2]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_jaccard_long(feature_df, n_items=5):
    result_df = feature_df
    result_df = result_df[result_df["is_daily"] == False]
    result_df = result_df[result_df["name_n_words"] > 2]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_jaccard_daily(feature_df, n_items=5):
    result_df = feature_df
    result_df = result_df[result_df["is_daily"] == True]
    print(result_df.shape)
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_geodesic(query_note, n_items=10):
    result_df = obsfeatures.geodesic_distances(query_note, vault.graph)
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df.head(n_items)

    return _df_to_items(result_df, lambda row: row.distance)


def title_item(title: str) -> dict:
    return {"name": f"🟦🟦🟦🟦🟦 {title} 🟦🟦🟦🟦🟦"}


@app.post("/similar")
def similar(request: ObsidianPyLabRequest):

    query_note_name = os.path.splitext(request.notePath)[0]
    query_note = [n for n in notes if n.name == query_note_name][0]

    items = []

    feature_df = _features_merged(query_note, vault.graph)
    items.append(title_item("Short"))
    items += get_items_jaccard_short(feature_df)

    items.append(title_item("Long"))
    items += get_items_jaccard_long(feature_df)

    items.append(title_item("Daily"))
    items += get_items_jaccard_daily(feature_df)

    return {"contents": items}
