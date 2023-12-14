#!/usr/bin/env python3
"""
Cosine Similiarity Matching
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from ..config import field_config

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer


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