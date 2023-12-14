#!/usr/bin/env python3
"""
Exact Matching
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from ..config import field_config

import pandas as pd
import numpy as np


def exact_matches(
        field_values: pd.Series, pairs: np.ndarray
        ) -> np.ndarray:
    """
    Performing exact matches on pairs
    """

    arr1 = field_values.loc[pairs[:, 0]].values
    arr2 = field_values.loc[pairs[:, 1]].values
    return ((arr1 == arr2) & (~pd.isna(arr1)) & (~pd.isna(arr2))).astype(int)