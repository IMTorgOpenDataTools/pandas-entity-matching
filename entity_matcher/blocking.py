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

    blocks = defaultdict(list)
    for idx, key in enumerate(field_values):
        if key is not None:
            blocks[key].append(idx)

    return blocks


stop_words_for_token_blocks = set(stopwords.words('english') + list(string.punctuation))

def token_blocking(df: pd.DataFrame, stop_words: set) -> dict[str, list]:

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

    sorted_indices = (
        df[keys].dropna(how="all").sort_values(keys).index.tolist()
    )
    pairs = []
    for window_end in range(1, len(sorted_indices)):
        window_start = max(0, window_end - window_size)
        for i in range(window_start, window_end):
            pairs.append([sorted_indices[i], sorted_indices[window_end]])

    return np.array(pairs)