#!/usr/bin/env python3
"""
Test Main
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from entity_matcher.preprocess import preprocess_df_columns
from tests.setup import prepare_dataset

import pandas as pd

from pathlib import Path


def test_preprocess():
    df = prepare_dataset(preprocess=False)

    #preprocess
    pp_df = preprocess_df_columns(df, cols=["title", "artist", "album"])
    
    assert pp_df.shape == (127451, 12)
