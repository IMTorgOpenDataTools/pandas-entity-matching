#!/usr/bin/env python3
"""
Levenshtein Distance Matching
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from ..config import field_config

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


def levenshtein_distance(
        field_values: pd.Series, pairs: np.ndarray
        ) -> np.ndarray:
    """
    Performing levenshtein distance matching on pairs

    Notes:
        - the range of ratios is 0 - 100
    """
    results = []
    previous_target = None
    for target in np.unique(pairs[:, 0]):
        if target != previous_target:
            comparison_index = [idx[1] for idx in pairs if idx[0]==target]
            comparison_values = field_values[comparison_index].values
            target_value = field_values.loc[target]
            distances = process.extract(target_value, 
                                        comparison_values, 
                                        scorer=fuzz.QRatio, 
                                        limit=None
                                        )
            ordered_distances = []
            for idx, compare_idx in enumerate(comparison_index):
                dist = [distance[1] for distance in distances if distance[2]==idx][0] / 100
                ordered_distances.append(dist)
            results.extend(ordered_distances)
            previous_target = target

    return np.array(results)