#!/usr/bin/env python3
"""
Matching
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from .config import field_config

import pandas as pd
import numpy as np

import itertools
from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer


def get_field_similarity_scores_from_pairs(
    df: pd.DataFrame, pairs: np.ndarray, field_config: dict[str, str]
) -> dict[str, np.ndarray]:
    """
    Measuring similarity by field. It is either cosine similarity
    (if sim_type == 'fuzzy') or exact match 0/1 (if sim_type == 'exact').
    Attribute's similarity scores are stored in field_score dictionary
    with the field name as key.

    Usage::
        >>> idx = 10
        >>> idx_group_1 = np.where(sb_pairs[:,0]==idx)
        >>> sb_pairs[np.where(sb_pairs[:,0]==idx)]
            array([[  10, 3113],
                    [  10, 5068],
                    [  10, 7786]], dtype=int32)
        >>>
        >>> sb_pairs[np.where(sb_pairs[:,0]==idx)]
        >>> rows_group_1 = sb_pairs[idx_group_1][:,1]
        >>> rows_group_1 = np.insert(rows_group_1, 0, idx)
        >>> row_pairs = df[['title','artist','album']].iloc[rows_group_1]
        >>>
        >>> scores = get_field_similarity_scores_from_pairs(df, sb_pairs, field_config)
        >>> scores['title'].shape
            (13794,)
    """

    field_scores = {}
    for field, sim_type in field_config.items():
        if sim_type == "fuzzy":
            mod_df = df[field].fillna("")
            field_scores[field] = cosine_similarities(mod_df, pairs)
        else:
            field_scores[field] = exact_matches(df[field], pairs)
    return field_scores


def cosine_similarities(
    field_values: pd.Series, pairs: np.ndarray
) -> np.ndarray:
    """
    Computing cosine similarities on pairs
    """
    token_matrix_1, token_matrix_2 = get_token_matrix_pair(field_values, pairs)
    cos_sim = cosine_similarities_on_pair_matrices(token_matrix_1, token_matrix_2)
    return cos_sim


def get_token_matrix_pair(
    field_values: pd.Series, pairs: np.ndarray,
) -> tuple[csr_matrix, csr_matrix]:
    """
    Converting pairs into matrices of token counts (matrix of records
    by tokens filled with token counts).
    """
    all_idx = np.unique(pairs)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3))
    vectorizer.fit(field_values.loc[all_idx])
    token_matrix_1 = vectorizer.transform(field_values.loc[pairs[:, 0]])
    token_matrix_2 = vectorizer.transform(field_values.loc[pairs[:, 1]])
    return token_matrix_1, token_matrix_2


def cosine_similarities_on_pair_matrices(
    token_matrix_1: csr_matrix, token_matrix_2: csr_matrix
) -> np.ndarray:
    """
    Computing cosine similarities on pair of token count matrices.
    It normalizes each record (axis=1) first, then computes dot product
    for each pair of records.
    """
    token_matrix_1 = normalize(token_matrix_1, axis=1)
    token_matrix_2 = normalize(token_matrix_2, axis=1)
    cos_sim = np.asarray(
        token_matrix_1.multiply(token_matrix_2).sum(axis=1)
    ).flatten()
    return cos_sim


def exact_matches(
    field_values: pd.Series, pairs: np.ndarray
) -> np.ndarray:
    """
    Performing exact matches on pairs
    """

    arr1 = field_values.loc[pairs[:, 0]].values
    arr2 = field_values.loc[pairs[:, 1]].values
    return ((arr1 == arr2) & (~pd.isna(arr1)) & (~pd.isna(arr2))).astype(int)


def calc_overall_scores(field_scores: dict[str, np.ndarray]) -> np.ndarray:
    return np.array(list(field_scores.values())).mean(axis=0)


def find_matches(scores: np.ndarray, threshold: float) -> np.ndarray:
    return scores >= threshold