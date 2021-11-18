import logging
import os
import pathlib
from typing import Optional

import numpy as np
import obsidiantools.api as otools
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware, record_timing
from pydantic import BaseModel

import obsfeatures
import vault_index


class ObsidianPyLabRequest(BaseModel):
    vaultPath: str
    notePath: Optional[str] = None
    text: Optional[str] = None


# Load .env
load_dotenv()

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_timing_middleware(app, record=logger.info, prefix="app", exclude="untimed")

# Available endpoints
available_fns = ["related", "reindex"]

# Read vault and notes into memory
vault_path = pathlib.Path(os.getenv("VAULT_PATH"))
index = vault_index.VaultIndex(vault_path)


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


def _features_merged(query_note, index: vault_index.VaultIndex) -> pd.DataFrame:
    jaccard_df = obsfeatures.jaccard_coefficients(query_note, index)
    note_individual_features = obsfeatures.get_notes_individual_df(index)
    geodesic_distances = obsfeatures.geodesic_distances(query_note, index.vault.graph)
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
    if name not in index.notes:
        # Try reloading vault
        index.load()

    return index.notes[name]


@app.post("/reindex")
def reindex(request: ObsidianPyLabRequest):
    index.load()
    return {"status": "ok"}


@app.post("/related")
def related(request: ObsidianPyLabRequest):

    query_note_name = os.path.splitext(request.notePath)[0]
    query_note = get_note_by_name(query_note_name)

    items = []

    feature_df = _features_merged(query_note, index)

    items.append(title_item("Long"))
    items += get_items_jaccard_long(feature_df, n_items=8)

    items.append(title_item("Short"))
    items += get_items_jaccard_short(feature_df, n_items=8)

    items.append(title_item("Daily"))
    items += get_items_jaccard_daily(feature_df)

    items.append(title_item("Nonexistent"))
    items += get_items_jaccard_nonexistent(feature_df)

    return {"contents": items}
