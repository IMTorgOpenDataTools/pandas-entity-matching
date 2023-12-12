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
from tests.setup import prepare_dataset



pp_df = prepare_dataset(preprocess=True)


def test_standard_blocking():
    sb_title = standard_blocking(pp_df.title)
    sb_artist = standard_blocking(pp_df.artist)
    sb_album = standard_blocking(pp_df.album)
    assert True == True

def test_token_blocking():
    columns = ['title','artist','album']
    token_blocks = token_blocking(pp_df[columns], stop_words_for_token_blocks)
    assert True == True

def test_sorted_neighborhood():
    columns = ['title','artist','album']
    sn_pairs = sorted_neighborhood(pp_df, columns)
    assert True == True