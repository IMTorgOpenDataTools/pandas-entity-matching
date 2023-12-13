"""
Test Blocking
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from entity_matcher.blocking import (
    standard_blocking,
    stop_words_for_token_blocks, 
    token_blocking, 
    sorted_neighborhood
    )
from entity_matcher.block_processing import (
    purge_blocks,
    get_pairs_from_blocks,

    get_adjacency_matrix_from_pairs,
    prune_edges,
    get_pairs_from_adj_matrix,

    get_union_of_adj_matrices,

    run_all_blocking_approaches
    )
from tests.setup import prepare_dataset

import pandas as pd


#setup
pp_df = prepare_dataset(preprocess=True)



def test_block_purging():
    columns = ['title','artist','album']

    sb_title = standard_blocking(pp_df.title)
    sb_artist = standard_blocking(pp_df.artist)
    sb_album = standard_blocking(pp_df.album)
    token_blocks = token_blocking(pp_df[columns], stop_words_for_token_blocks)

    token_blocks = purge_blocks(token_blocks)
    sb_title = purge_blocks(sb_title)
    sb_artist = purge_blocks(sb_artist)
    sb_album = purge_blocks(sb_album)
    assert len(sb_title) == 9862


def test_meta_blocking():
    columns = ['title','artist','album']
    samp_df = pp_df.sample(frac=.10, random_state=1)
    token_blocks = token_blocking(samp_df[columns], stop_words_for_token_blocks)
    pairs = get_pairs_from_blocks(token_blocks)
    adj_matrix = get_adjacency_matrix_from_pairs(pairs, (len(pp_df), len(pp_df)))
    adj_matrix = prune_edges(adj_matrix, edge_weight_threshold=2, verbose=True)
    tb_pairs = get_pairs_from_adj_matrix(adj_matrix)
    assert tb_pairs.shape == (169388, 2)


def test_union_of_blocks():
    samp_df = pp_df.sample(frac=.10, random_state=1)
    sb_title = standard_blocking(samp_df.title)
    sb_artist = standard_blocking(samp_df.artist)
    sb_album = standard_blocking(samp_df.album)

    adj_matrix_list = []
    for blocks in [sb_title, sb_artist, sb_album]:
        pairs = get_pairs_from_blocks(blocks)
        adj_matrix_list.append(
            get_adjacency_matrix_from_pairs(pairs, (len(samp_df), len(samp_df)))
        )
    adj_matrix_union = get_union_of_adj_matrices(adj_matrix_list)
    sb_pairs = get_pairs_from_adj_matrix(adj_matrix_union)

    assert sb_pairs.shape == (6867135, 2)


def test_all_blocking_approaches():
    #TODO
    result = run_all_blocking_approaches(df=pp_df, cols=['title','artist','album'])
    pd.DataFrame(
        [
            ["Standard Blocking", f"{len(result['sb_pairs']):,}"],
            ["Token Blocking", f"{len(result['tb_pairs']):,}"],
            ["Sorted Neighborhood", f"{len(result['sn_pairs']):,}"],
        ],
        columns=["Blocking Approach", "N of Candidate Pairs"]
    )
    assert True == True