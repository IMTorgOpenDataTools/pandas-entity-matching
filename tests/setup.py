"""
Prepare data
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from entity_matcher.preprocess import preprocess_df_columns

import pandas as pd

from pathlib import Path


def prepare_dataset(preprocess=True):
    filename = 'data.csv'
    filepath = './tests/data' / Path(filename)
    raw = pd.read_csv(filepath)

    #duplicates (same CID)
    assert raw[raw.CID == 41689].shape[0] == 3

    #english songs
    english_cids = raw[
                        raw.language.str.lower()
                        .str.contains("^en|^eg", na=False)
                        ].CID.unique()
    df = raw[raw.CID.isin(english_cids)].reset_index(drop=True)
    assert df.shape[0] > 0

    #preprocess
    if preprocess:
        df = preprocess_df_columns(df, cols=["title", "artist", "album"])

    return df