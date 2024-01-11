#!/usr/bin/env python3
"""
Review matching results
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

import pandas as pd




def get_similar_text(df, target_column, text, match_column):
    """Get all records that contain specified text, then compare with match results

    Usage::
        >>> df = Matcher.preprocess(df)
        >>> df['Proposed matches'] = Matcher.get_matches(df)
        >>> get_similar_text(df, target_column, text, 'Proposed matches')
    
    """
    records = df[df[target_column].str.lower().str.contains(text, na=False)]
    if records.shape[0]==0:
        print(f'ERROR: text f{text} could not be found')
        raise KeyError
    match_ids = records[match_column].unique().tolist()
    match_ids.remove('')
    match_df = df[df[match_column].isin(match_ids)]
    result_df = pd.concat([records, match_df])
    return result_df
