#!/usr/bin/env python3
"""
EntityMatcher class used for entrypoint to the module.
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from .config import (
    field_config,
    validate_config
)
from .blocking import (
    standard_blocking,
    token_blocking
)
from .block_processing import (
    purge_blocks,
    meta_blocks
)
from . import preprocess
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
    _available_blocking_methods = ['standard', 'purge']
    _available_scoring_methods = ['fuzzy', 'exact']
    _default_combined_pair_name = 'Combined_Fields'
    _default_blocking = {"operation": "standard", "process": "purge"}      


    def __init__(self, field_config: dict[str, str]) -> None:
        validated_shape = validate_config(field_config)
        if validated_shape:
            if field_config['blocking'] !={}:
                for method in field_config['blocking'].values():
                    if method not in self._available_blocking_methods:
                        raise TypeError
            for method in field_config['scoring'].values():
                if method not in self._available_scoring_methods:
                    raise TypeError
        self.field_config = field_config


    def preprocess(self, df: pd.DataFrame, columns: list=None) -> pd.DataFrame:
        """Prepare df columns for matching."""
        if not columns:
            columns = list( self.field_config['scoring'].keys() )
        for column in columns:
            if column not in df.columns:
                print(f'Column ${column} not in dataframe')
                raise TypeError
        return preprocess.preprocess_df_columns(df, columns)


    def get_matches(self, df: pd.DataFrame) -> np.ndarray:
        """Get a column of matches, aligned to the dataframe, based on the `field_config`.
        
        """
        
        #check memory requirement and applying default blocking if necessary
        if not self.field_config['blocking']:
            for field in self.field_config['scoring']:
                if self.is_process_memory_constrained(df[field])['dot_product_size']:
                    print(f'''INFO: Records for field {field}, are too many for memory 
                          constraints at: {df[field].shape[0]} records!  Applying blocking.'''
                          )
                    self.field_config['blocking'] = self._default_blocking

        #create pairs from indices
        pair_dict = self.create_pairs_from_blocking_approach(
            df=df,
            field_config=self.field_config
        )
            
        #score pairs based on configuration methods
        '''
        pair_dict
        {'title': array([[ 1,  0],
           ...[30, 29]])}'''
        matches = self.get_field_similarity_scores_from_pairs(
            df=df, 
            pair_dict=pair_dict, 
            field_config=self.field_config,
            threshold=self._threshold
            )
        
        #align results with original index
        groups = self.match_pairs_to_columns(
            df=df, 
            matches=matches
            )

        return groups


    def is_process_memory_constrained(self, arr: np.array) -> bool:
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
        tolerance = records_size_in_mb - (arr.nbytes / 1024 ** 2)
        assert tolerance < .0001
        dot_product_size_in_mb = item_size * arr.shape[0] * arr.shape[0] / 1024 ** 2

        if records_size_in_mb < self._available_memory:
            result['record_size'] = False
        if dot_product_size_in_mb < self._available_memory:
            result['dot_product_size'] = False

        return result
    

    def create_unique_index_pairs(self, df):
        """Create pairs for each record index when not applying blocking."""
        rows = df.shape[0] - 1
        groups = []
        for idx in range(0,rows):
            group = [[idx, item] for item in range(0,rows) if not (idx <= item)]
            groups.extend(group)
        pairs = np.array(groups)
        return pairs
    

    def create_pairs_from_blocking_approach(self, df: pd.DataFrame, field_config: dict) -> dict[str, np.ndarray]:
        """Pairs are generated based on blocking operation and process methods,
        if methods are chosen.

        Note: This method forces a consistent output which may not be possible 
        for some blocking approaches, such as `sorted_neighborhood()`.
        
        """
        pairs = {}
        fields = field_config['scoring'].keys()

        if field_config['blocking'] == {}:
            for field in fields:
                pairs[field] = self.create_unique_index_pairs(df[field])

        else:
            op = field_config['blocking']['operation']
            match op:
                case 'standard':
                    for field in fields:
                        pairs[field] = standard_blocking(df[field])
                #case 'token':
                #    for field in fields:
                #        pairs[field] = token_blocking(df[field])
            proc = field_config['blocking']['process']
            match proc:
                case 'purge':
                    for field in fields:
                        dicts = purge_blocks(pairs[field])
                        dicts_without_nan = {k:v for k,v in dicts.items() if k is not np.nan}
                        indices = np.array( [np.array(pair) for pair in dicts_without_nan.values()], dtype="object" )
                        tuples = []
                        for idx, line in enumerate(indices):
                            grp = []
                            if len(line) > 2:
                                grp.append(line[:2])
                                for item in line[2:]:
                                    grp.append(np.append(line[0], item))
                            else:
                                grp.append(line)
                            #print(idx)
                            tuples.extend(grp)
                        pairs[field] = np.array(tuples)
                    
        return pairs
    

    def get_field_similarity_scores_from_pairs(
            self,
            df: pd.DataFrame, pair_dict: dict[str, np.ndarray], field_config: dict[str, str], threshold: float
            ) -> pd.DataFrame:
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
        matches = pd.DataFrame()

        #get score for each field
        field_scores = {}
        for field, sim_type in field_config['scoring'].items():
            pairs = pair_dict[field]
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
                    field_scores[field] = embedding.wv_cosine_similarities(mod_df, pairs)       #TODO
        
        fields = list( pair_dict.keys() )
        #combined score across fields
        if fields == [self._default_combined_pair_name]:
            pass

        #individual score for each field
        elif all( [field in df.columns for field in fields] ):
            tmpDf = pd.DataFrame()
            for field in fields:
                tmp = pd.DataFrame(pair_dict[field])
                tmp[field] = field_scores[field]
                if tmpDf.empty:
                    tmpDf = tmp
                else: 
                    tmpDf = pd.merge(tmpDf, tmp, on=[0,1], how='outer')
            tmpDf['scores'] = tmpDf[fields].mean(axis=1, skipna=True)
            matches = tmpDf[tmpDf['scores']>threshold].reset_index(drop=True)
            assert matches[0].unique().shape[0] <= matches[0].shape[0]

        #some error occured
        else:
            raise TypeError
        
        #ensure blocking indexes are maintained
        #TODO:assert scores.index == pd.index

        return matches
    

    def match_pairs_to_columns(
            self, 
            df: pd.DataFrame, matches: pd.DataFrame, new_column_name: str ='group'
            ) -> np.ndarray:
        """..

        sb_pairs[np.where(sb_pairs[:,0]==idx)]
        rows_group_1 = sb_pairs[idx_group_1][:,1]
        rows_group_1 = np.insert(rows_group_1, 0, idx)
        row_pairs = df[['title','artist','album']].iloc[rows_group_1]
        
        scores = get_field_similarity_scores_from_pairs(df, sb_pairs, field_config)
        scores['title'].shape
        (13794,).
        
        TODO:optimize performance 
        """

        df[new_column_name] = ''
        for i in matches.index:
            #print( df['title'].iloc[ list(matches[[0,1]].iloc[i]) ] )
            grp = list(matches[[0,1]].iloc[i])
            if df[new_column_name].iloc[grp[0]] == '':
                rows = matches[matches[0].isin(grp)]
                idxs = list(set(rows[1].values))
                idxs.insert(0, grp[0])
                df[new_column_name].iloc[ idxs ] = grp[0]
        new_column = df[new_column_name]
        df.drop(columns=[new_column_name], inplace=True)
        return new_column

        


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