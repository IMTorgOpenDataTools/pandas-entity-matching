#!/usr/bin/env python3
"""
Blocking
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from collections import defaultdict

import pandas as pd
import numpy as np

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


def standard_blocking(field_values: pd.Series) -> dict[str, list]:
    """Each value has an index of rows it is used.
    
    Usage::
        >>> sb_title = standard_blocking(pp_df.title)
        >>> key = list(sb_title.keys())[5]
        >>> sb_title[key]
            'macabre discovery distorted drama': [5, 109011]
    """
    blocks = defaultdict(list)
    for idx, key in enumerate(field_values):
        if key is not None:
            blocks[key].append(idx)

    return blocks


stop_words_for_token_blocks = set(stopwords.words('english') + list(string.punctuation))

def token_blocking(df: pd.DataFrame, stop_words: set) -> dict[str, list]:
    """Each token (of the row of concatenated fields) has an index of rows it is used.

    Usage::
        >>> token_blocks = token_blocking(pp_df[columns], stop_words_for_token_blocks)
        >>> key = list(token_blocks.keys())[0]
        >>> token_blocks[key]
            [0, 1322, 2024, 2522, 3140, 3567, 4838, 8641, 8808, 12284, 13095, 15137, 16321, 18281, ...]
        >>> tmp = list(token_blocks.values())[0]
        >>> pp_df[columns].iloc[tmp]
    """
    blocks = defaultdict(list)
    for i, row in enumerate(df.itertuples()):
        # concatenate columns and tokenize
        string = " ".join([str(value) for value in row if not pd.isna(value)])
        tokens = set(
            [word for word in word_tokenize(string) if word not in stop_words]
        )
        # create blocks
        for token in tokens:
            blocks[token].append(i)

    return blocks


def sorted_neighborhood(
    df: pd.DataFrame, keys: list, window_size: int = 3
) -> np.ndarray:
    """Pairs of rows created from the sorted columns

    Usage::
        >>> sn_pairs = sorted_neighborhood(pp_df, columns)
        >>> sn_pairs
            array([[ 99353,  13878],
                   [ 99353,  73335],
                   [ 13878,  73335],
                   ...,
                   [ 59927,  20462],
                   [127393,  20462],
                   [ 90786,  20462]])
        >>> pp_df[columns].iloc[sn_pairs[3]]
    """

    sorted_indices = (
        df[keys].dropna(how="all").sort_values(keys).index.tolist()
    )
    pairs = []
    for window_end in range(1, len(sorted_indices)):
        window_start = max(0, window_end - window_size)
        for i in range(window_start, window_end):
            pairs.append([sorted_indices[i], sorted_indices[window_end]])

    return np.array(pairs)