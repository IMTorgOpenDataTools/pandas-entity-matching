#!/usr/bin/env python3
"""
Summarize dataframe and results
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"



def df_sparsity_summary(df, xcol, ycols=[]):
    """Given a dataframe and a column of categorical variables, get a summary of
    how many records are filled / missing for other column(s).

    Usage::
        >>> ssum_df = df_sparsity_summary(df, xcol, ycols=[])
        >>> ssum_df
                        title  language  idx
            SourceID                      
            1             6         0    6
            2             9         9    9
            3             7         7    7
            4             6         6    6
            5             4         4    4
        Note: a count of '0' implies all values are of 'NaN'
    """
    new_index_name = 'total'
    df.reset_index(inplace=True)
    df.rename(columns={'level_0': new_index_name}, inplace=True)
    ycols.append(new_index_name)
    result_df = df.groupby([xcol])[ycols].count()
    return result_df
        


def create_sparsity_table(df):
    """Create a nicely formatted table from the results of `df_sparsity_summary`
    
    Usage::
        >>> ssum_df = df_sparsity_summary(df, xcol, ycols=[])
        >>> create_sparsity_table(ssum_df)   
    """
    return True