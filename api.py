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
available_fns = ["related"]

# Read vault and notes into memory
def load_vault():
    VAULT_PATH = "/Users/taivo/kb"
    vault = otools.Vault(pathlib.Path(VAULT_PATH)).connect()
    notes = [
        obsfeatures.Note.from_path(name, VAULT_PATH / p)
        for name, p in vault.file_index.items()
    ]
    print(f"{len(notes)} notes in vault")

    return vault, notes


vault, notes = load_vault()


@app.get("/")
def read_root():
    return {"scripts": [f"http://127.0.0.1:5000/function/{fn}" for fn in available_fns]}


def _row_to_dict(row):
    res = {}
    for k, v in row.items():
        if k == "name":
            continue
        if np.isinf(v) or np.isnan(v):
            res[k] = None
        else:
            res[k] = v

    return res


def _df_to_items(df: pd.DataFrame, score_getter):
    items = []

    for _, row in df.iterrows():
        items.append(
            {
                "path": f'{row["name"]}.md',
                "name": row["name"],
                "info": {"score": score_getter(row), "features": _row_to_dict(row)},
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
    note_individual_features = obsfeatures.get_notes_individual_df(notes, vault)
    geodesic_distances = obsfeatures.geodesic_distances(query_note, vault.graph)
    df = jaccard_df.merge(note_individual_features, on="name", how="left")
    df = df.merge(geodesic_distances, on="name", how="left")

    return df


def get_items_jaccard_short(feature_df, n_items=5):
    result_df = feature_df
    result_df = result_df[result_df["is_daily"] == False]
    result_df = result_df[result_df["name_n_words"] <= 2]
    result_df = result_df[result_df["plaintext_n_words"] >= 10]
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df[result_df["jaccard"] > 0.0]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_jaccard_long(feature_df, n_items=5):
    result_df = feature_df
    result_df = result_df[result_df["is_daily"] == False]
    result_df = result_df[result_df["name_n_words"] > 2]
    result_df = result_df[result_df["plaintext_n_words"] >= 10]
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df[result_df["jaccard"] > 0.0]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_jaccard_daily(feature_df, n_items=5):
    result_df = feature_df
    result_df = result_df[result_df["is_daily"] == True]
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df[result_df["jaccard"] > 0.0]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_jaccard_nonexistent(feature_df, n_items=5):
    result_df = feature_df
    result_df = result_df[
        np.logical_or(result_df["exists"] != True, result_df["plaintext_n_words"] < 10)
    ]  # for nonexistent notes it will be None/nan, not False
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df[result_df["jaccard"] > 0.0]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def title_item(title: str) -> dict:
    return {"name": f"🟦 {title} 🟦"}


def get_note_by_name(name: str) -> obsfeatures.Note:
    global vault, notes
    results = [n for n in notes if n.name == name]
    if results:
        return results[0]
    else:
        # Try reloading vault
        vault, notes = load_vault()
        results2 = [n for n in notes if n.name == name]
        if results2:
            return results2[0]
        else:
            raise ValueError(f"No note with name {name}")


@app.post("/related")
def related(request: ObsidianPyLabRequest):

    query_note_name = os.path.splitext(request.notePath)[0]
    query_note = get_note_by_name(query_note_name)

    items = []

    feature_df = _features_merged(query_note, vault.graph)

    items.append(title_item("Long"))
    items += get_items_jaccard_long(feature_df, n_items=8)

    items.append(title_item("Short"))
    items += get_items_jaccard_short(feature_df, n_items=8)

    items.append(title_item("Daily"))
    items += get_items_jaccard_daily(feature_df)

    items.append(title_item("Nonexistent"))
    items += get_items_jaccard_nonexistent(feature_df)

    return {"contents": items}
