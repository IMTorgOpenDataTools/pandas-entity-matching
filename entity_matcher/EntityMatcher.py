#!/usr/bin/env python3
"""
EntityMatcher class used for entrypoint to the module.
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from .config import field_config
from .matching import get_field_similarity_scores_from_pairs

import pandas as pd
import numpy as np

import itertools
from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer

import psutil



class EntityMatcher:
    """..."""

    _threshold = 0.70
    _available_memory = psutil.virtual_memory().free       #TODO:determine process based on resources
    _available_processes = ['cosine_similarity']

    def __init__(self, df: pd.DataFrame, field_config: dict[str, str]) -> None:
        self.df = df
        self.field_config = field_config

    def get_matches(self, process: str) -> np.ndarray:
        """..."""
        if process not in self._available_processes:
            raise TypeError
        
        '''
        for field in self.field_config:
            if self.df[field].memory_usage < self._available_memory:        #TODO:fix
        '''
        rows = self.df.shape[0] - 1
        groups = []
        for idx in range(0,rows):
            group = [[idx, item] for item in range(0,rows) if idx!=item]
            groups.extend(group)

        pairs = np.array(groups)
        scores = get_field_similarity_scores_from_pairs(df=self.df, pairs=pairs, field_config=self.field_config)
        groups = self.match_pairs_to_columns(pairs, scores, self._threshold)

        return groups
    
    def match_pairs_to_columns(self, pairs, scores, threshold, new_column_name='group'):
        """..

        sb_pairs[np.where(sb_pairs[:,0]==idx)]
        rows_group_1 = sb_pairs[idx_group_1][:,1]
        rows_group_1 = np.insert(rows_group_1, 0, idx)
        row_pairs = df[['title','artist','album']].iloc[rows_group_1]
        
        scores = get_field_similarity_scores_from_pairs(df, sb_pairs, field_config)
        scores['title'].shape
        (13794,)."""
        tmpDf = pd.DataFrame(pairs)
        tmpDf['scores'] = scores['title']
        matches = tmpDf[tmpDf['scores']>threshold].reset_index(drop=True)
        assert matches[0].unique().shape[0] <= matches[0].shape[0]

        self.df[new_column_name] = ''
        for i in matches.index:
            #print( df['title'].iloc[ list(matches[[0,1]].iloc[i]) ] )
            grp = list(matches[[0,1]].iloc[i])[0]
            if self.df[new_column_name].iloc[grp] == '':
                rows = matches[matches[0]==grp]
                idxs = list(rows[1])
                idxs.insert(0, grp)
                self.df[new_column_name].iloc[ idxs ] = grp

        return self.df[new_column_name]