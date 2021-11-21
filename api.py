import logging
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware
from pydantic import BaseModel

import features
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
vault_path = pathlib.Path(os.getenv("VAULT_PATH") or ".")
enable_top2vec = os.getenv("ENABLE_TOP2VEC") == "1"
index = vault_index.VaultIndex(vault_path, enable_top2vec=enable_top2vec)


def _row_to_dict(row: pd.Series) -> dict:
    res: Dict[str, Any] = dict()
    for k, v in row.items():
        if k == "name":
            continue
        if np.isinf(v) or np.isnan(v):
            res[k] = None
        else:
            res[k] = v

    return res


def _df_to_items(
    df: pd.DataFrame, score_getter: Callable[[pd.Series], float]
) -> List[Dict]:
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


def get_items_jaccard_short(feature_df: pd.DataFrame, n_items: int = 5):
    result_df = feature_df
    result_df = result_df[result_df["is_daily"] == False]
    result_df = result_df[result_df["name_n_words"] <= 2]
    result_df = result_df[result_df["plaintext_n_words"] >= 10]
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df[result_df["jaccard"] > 0.0]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_jaccard_long(feature_df: pd.DataFrame, n_items: int = 5):
    result_df = feature_df
    result_df = result_df[result_df["is_daily"] == False]
    result_df = result_df[result_df["name_n_words"] > 2]
    result_df = result_df[result_df["plaintext_n_words"] >= 10]
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df[result_df["jaccard"] > 0.0]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_jaccard_daily(feature_df: pd.DataFrame, n_items: int = 5):
    result_df = feature_df
    result_df = result_df[result_df["is_daily"] == True]
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df[result_df["jaccard"] > 0.0]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_jaccard_nonexistent(feature_df: pd.DataFrame, n_items: int = 5):
    result_df = feature_df
    result_df = result_df[
        np.logical_or(result_df["exists"] != True, result_df["plaintext_n_words"] < 10)
    ]  # for nonexistent notes it will be None/nan, not False
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df[result_df["jaccard"] > 0.0]
    result_df = result_df.sort_values("jaccard", ascending=False).head(n_items)

    return _df_to_items(result_df, lambda row: row.jaccard)


def get_items_top2vec(top2vec_df: pd.DataFrame, n_items: int = 5):
    result_df = top2vec_df
    result_df = result_df[result_df["is_daily"] == False]
    result_df = result_df[result_df["distance"] >= 2]
    result_df = result_df.sort_values("top2vec_similarity", ascending=False).head(
        n_items
    )

    return _df_to_items(result_df, lambda row: row.top2vec_similarity)


def make_title_item(title: str) -> dict:
    return {"name": f"🟦 {title} 🟦"}


def get_note_by_name(name: str) -> features.Note:
    if name not in index.notes:
        # Try reloading vault in case note was just created
        index.load()

    return index.notes[name]


@app.get("/")
def read_root():
    return {"scripts": [f"http://127.0.0.1:5000/function/{fn}" for fn in available_fns]}


@app.post("/reindex")
def reindex(request: ObsidianPyLabRequest):
    index.load(reindex_top2vec=True)
    return {"status": "ok"}


@app.post("/related")
def related(request: ObsidianPyLabRequest):

    query_note_name = os.path.splitext(request.notePath)[0]
    query_note = get_note_by_name(query_note_name)

    items = []

    feature_df = features.base_features(query_note, index)

    items.append(make_title_item("Long"))
    items += get_items_jaccard_long(feature_df, n_items=8)

    items.append(make_title_item("Short"))
    items += get_items_jaccard_short(feature_df, n_items=8)

    items.append(make_title_item("Daily"))
    items += get_items_jaccard_daily(feature_df)

    items.append(make_title_item("Nonexistent"))
    items += get_items_jaccard_nonexistent(feature_df)

    if index.enable_top2vec and query_note.name in index.top2vec_model.document_ids:
        top2vec_df = features.top2vec_features(query_note, index).merge(
            feature_df, on="name", how="left"
        )
        items.append(make_title_item("Top2Vec"))
        items += get_items_top2vec(top2vec_df, n_items=10)

    return {"contents": items}
