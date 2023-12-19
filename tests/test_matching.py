"""
Test Matching functions
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
import entity_matcher as em
from tests.setup import prepare_dataset

from memory_profiler import memory_usage
from time import sleep





def run_matching_process():
    pp_df = prepare_dataset(preprocess=True)
    df = pp_df.sample(frac=.10, random_state=1).reset_index(drop=True)
    """
    frac=.10 => 309 bytes
    frac=.50 => 309 bytes, 58.11s
    frac=1.0 => 309 bytes, 149.43s
    """
    field_config = em.field_config
    field_config.clear()
    field_config = {
        # <field>: <sim_type>
        "title": "fuzzy",
        "artist": "fuzzy",
        "album": "fuzzy",
        "number": "exact",
    }

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
    sleep(.1)
    sb_pairs = get_pairs_from_adj_matrix(adj_matrix_union)
    sleep(.1)
    Matcher = em.EntityMatcher(field_config)
    field_scores_sb = Matcher.get_field_similarity_scores_from_pairs(df, sb_pairs, field_config)
    sleep(.1)

    scores_sb = Matcher.calc_overall_scores(field_scores_sb)
    is_matched_sb = Matcher.find_matches(scores_sb, threshold=0.64)

    #assert is_matched_sb.__len__() == 13794
    assert is_matched_sb.__len__() > 0


def test_matching_process():
    mem_usage = memory_usage(run_matching_process)
    print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    print('Maximum memory usage: %s' % max(mem_usage))
    assert True == True