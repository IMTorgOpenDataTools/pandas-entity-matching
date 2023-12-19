#!/usr/bin/env python3
"""
EntityMatcher class used for entrypoint to the module.
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from .config import field_config
from .matching import (
    cosine,
    exact,
    levenshtein,
    embedding
)

import pandas as pd
import numpy as np

import psutil



class EntityMatcher:
    """..."""

    _threshold = 0.70
    _available_memory = psutil.Process().memory_info().rss          #largest physical memory process used; convert to MB rss / 1024 ** 2
    _available_methods = ['fuzzy', 'exact']


    def __init__(self, field_config: dict[str, str]) -> None:
        for method in field_config.values():
            if method not in self._available_methods:
                raise TypeError
        self.field_config = field_config

    
    def is_memory_constrained(self, arr: np.array) -> bool:
        """Determine if the dot product is constrained by memory.
        Assuming default datatype `float64` with 8 bytes.
        ref: https://stackoverflow.com/questions/25046076/python-numpy-huge-matrices-multiplication
        """
        result = {
            'record_size': True,
            'dot_product_size': True
        }
        item_size = arr.dtype.itemsize
        records_size_in_mb = item_size * arr.shape[0] * 1 / 1024 ** 2
        assert records_size_in_mb == arr.nbytes
        dot_product_size_in_mb = item_size * arr.shape[0] * arr.shape[0] / 1024 ** 2

        if records_size_in_mb < self._available_memory:
            result['record_size'] = False
        if dot_product_size_in_mb < self._available_memory:
            result['dot_product_size'] = False

        return result


    def get_matches(self, df: pd.DataFrame, blocking: bool=False) -> np.ndarray:
        """..."""
        
        #check memory requirement
        if not blocking:
            for field in self.field_config:
                if self.is_memory_constrained(df[field])['dot_product_size']:
                    blocking = True      

        #use blocking if not enough memory
        if not blocking:
            rows = df.shape[0] - 1
            groups = []
            for idx in range(0,rows):
                group = [[idx, item] for item in range(0,rows) if not (idx <= item)]
                groups.extend(group)
            pairs = np.array(groups)

        scores = self.get_field_similarity_scores_from_pairs(df=df, pairs=pairs, field_config=self.field_config)
        groups = self.match_pairs_to_columns(df, pairs, scores, self._threshold)

        return groups
    

    def match_pairs_to_columns(
            self, 
            df: pd.DataFrame, pairs: np.ndarray, scores: np.ndarray, threshold: float, new_column_name: str ='group'
            ) -> np.ndarray:
        """..

        sb_pairs[np.where(sb_pairs[:,0]==idx)]
        rows_group_1 = sb_pairs[idx_group_1][:,1]
        rows_group_1 = np.insert(rows_group_1, 0, idx)
        row_pairs = df[['title','artist','album']].iloc[rows_group_1]
        
        scores = get_field_similarity_scores_from_pairs(df, sb_pairs, field_config)
        scores['title'].shape
        (13794,)."""
        tmpDf = pd.DataFrame(pairs)
        #TODO:account for multiple fields, here, not just `title`
        tmpDf['scores'] = scores['title']
        matches = tmpDf[tmpDf['scores']>threshold].reset_index(drop=True)
        assert matches[0].unique().shape[0] <= matches[0].shape[0]

        df[new_column_name] = ''
        for i in matches.index:
            #print( df['title'].iloc[ list(matches[[0,1]].iloc[i]) ] )
            grp = list(matches[[0,1]].iloc[i])[0]
            if df[new_column_name].iloc[grp] == '':
                rows = matches[matches[0]==grp]
                idxs = list(rows[1])
                idxs.insert(0, grp)
                df[new_column_name].iloc[ idxs ] = grp

        return df[new_column_name]
    

    def get_field_similarity_scores_from_pairs(
            self,
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
            match sim_type:
                case "exact":
                    field_scores[field] = exact.exact_matches(df[field], pairs)
                case "fuzzy": 
                    mod_df = df[field].fillna("")
                    field_scores[field] = cosine.cosine_similarities(mod_df, pairs)
                case "levenshtein":
                    mod_df = df[field].fillna("")
                    field_scores[field] = levenshtein.levenshtein_distance(mod_df, pairs)       #TODO
                case "embedding":
                    mod_df = df[field].fillna("")
                    field_scores[field] = embedding.nn_cosine_similarities(mod_df, pairs)       #TODO

        return field_scores


    def calc_overall_scores(
            self,
            field_scores: dict[str, np.ndarray]
            ) -> np.ndarray:
        """..."""
        return np.array(list(field_scores.values())).mean(axis=0)


    def find_matches(
            self,
            scores: np.ndarray, threshold: float
            ) -> np.ndarray:
        """..."""
        return scores >= threshold