"""
Test All
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


from entity_matcher.blocking import (
    standard_blocking, 
    token_blocking, 
    sorted_neighborhood
    )
from entity_matcher.block_processing import (
    purge_blocks, 
    get_pairs_from_blocks, 
    get_adjacency_matrix_from_pairs, 
    get_union_of_adj_matrices, 
    get_pairs_from_adj_matrix
    )
from entity_matcher.matching import (
    get_field_similarity_scores, 
    calc_overall_scores, 
    find_matches,
    field_config
    )
from tests.setup import prepare_dataset



def test_all():
    df = prepare_dataset(preprocess=True)

    sb_title = standard_blocking(df.title)
    sb_artist = standard_blocking(df.artist)
    sb_album = standard_blocking(df.album)

    sb_title = purge_blocks(sb_title)
    sb_artist = purge_blocks(sb_artist)
    sb_album = purge_blocks(sb_album)

    adj_matrix_list = []
    for blocks in [sb_title, sb_artist, sb_album]:
        pairs = get_pairs_from_blocks(blocks)
        adj_matrix_list.append(
            get_adjacency_matrix_from_pairs(pairs, (len(df), len(df)))
        )
    adj_matrix_union = get_union_of_adj_matrices(adj_matrix_list)
    sb_pairs = get_pairs_from_adj_matrix(adj_matrix_union)
    field_scores_sb = get_field_similarity_scores(df, sb_pairs, field_config)

    scores_sb = calc_overall_scores(field_scores_sb)
    is_matched_sb = find_matches(scores_sb, threshold=0.64)

    assert is_matched_sb.__len__() == 1338965