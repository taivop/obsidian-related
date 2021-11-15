import re

import networkx as nx
import numpy as np
import pandas as pd

import obsfeatures


def exclude_below_distance(
    df: pd.DataFrame, graph: nx.Graph, source_note: obsfeatures.Note, max_distance: int
):
    """Exclude all notes above max_distance from source_note."""
    res = df.copy()
    res = res.merge(obsfeatures.geodesic_distances(source_note, graph), on="name")
    res = res[res["distance"] <= max_distance]
    return res


def short_names(df: pd.DataFrame) -> pd.DataFrame:
    res = df[df["is_daily"] == False]
    res = res.sort_values("name_n_words")

    return res


def long_names(df: pd.DataFrame) -> pd.DataFrame:
    res = df[df["is_daily"] == False]
    res = res.sort_values("name_n_words", ascending=False)
    return res


def substantive_daily_notes(df: pd.DataFrame) -> pd.DataFrame:
    res = df[df["is_daily"] == True]
    n_words_threshold = np.quantile(res["plaintext_n_words"], 0.8)
    res = res[res["plaintext_n_words"] >= n_words_threshold]
    res = res.sort_values("plaintext_n_words", ascending=False)

    return res


def outlier_long_words(df: pd.DataFrame) -> pd.DataFrame:
    res = df[df["plaintext_n_words"] > 0].copy()
    res["chars_per_word"] = res["plaintext_n_char"] / res["plaintext_n_words"]
    chars_per_word_threshold = np.quantile(res["chars_per_word"], 0.9)
    n_words_threshold = np.quantile(res["plaintext_n_words"], 0.2)
    res = res[res["chars_per_word"] >= chars_per_word_threshold]
    res = res[res["plaintext_n_words"] >= n_words_threshold]
    res = res.sort_values("chars_per_word", ascending=False)

    return res
