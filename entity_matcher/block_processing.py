#!/usr/bin/env python3
"""
Block Processing
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


import pandas as pd
import numpy as np


def purge_blocks(
    blocks: dict[str, list], purging_threshold: int = 10000
) -> dict[str, list]:
    """Reduces the number of items in a block to a maximum threshold value

    Usage::
        >>> sb_title = purge_blocks(sb_title)
        >>> sb_title
            {'well you needn t first down': [2, 33833], 'macabre discovery di...
        >>> token_blocks = purge_blocks(token_blocks)
        >>> token_blocks
            ...
    """
    blocks_purged = {
        key: indices
        for key, indices in blocks.items()
        if len(indices) < purging_threshold and len(indices) > 1
    }
    return blocks_purged


import itertools
from scipy.sparse import csr_matrix

def meta_blocks(df: pd.DataFrame, blocks: dict[str, list], edge_weight_threshold: int = 2) -> np.ndarray:
    """Transform blocks into graph and prune lower-weighted edges

    """
    pairs = get_pairs_from_blocks(blocks)
    adj_matrix1 = get_adjacency_matrix_from_pairs(pairs, (len(df), len(df)))
    adj_matrix2 = prune_edges(adj_matrix1, edge_weight_threshold, verbose=True)
    new_pairs = get_pairs_from_adj_matrix(adj_matrix2)
    return new_pairs


def get_pairs_from_blocks(blocks: dict[str, list]) -> list[list]:
    """Transform list to get pair (a,b) from each combination of values 

    Usage::
        >>> key = list(token_blocks.keys())[0]
        >>> token_blocks[key]
            [0, 965, 1806, 6488, 8693, 9577, 11060, 12338, 12598]
        >>> pairs = get_pairs_from_blocks(token_blocks)
        >>> pairs
            [(0, 965), (0, 1806), (0, 6488), (0, 8693), (0, 9577), (0, 11060), (0, 12338), (0, 12598), (965, 1806), (965, 6488), (965, ...
        >>> len(token_blocks)
            35977
        >>> len(pairs)
            3102150
    """
    return [
        pair
        for indices in blocks.values()
        for pair in list(itertools.combinations(indices, 2))
    ]


def get_adjacency_matrix_from_pairs(
    pairs: list[list], matrix_shape: tuple[int, int]
) -> csr_matrix:
    """ """

    idx1 = [pair[0] for pair in pairs]
    idx2 = [pair[1] for pair in pairs]
    ones = np.ones(len(idx1))
    return csr_matrix(
        (ones, (idx1, idx2)), shape=matrix_shape, dtype=np.int8
    )


def prune_edges(
    adj_matrix: csr_matrix,
    edge_weight_threshold: float,
    verbose: bool = False,
) -> csr_matrix:
    """ """

    adj_matrix_pruned = adj_matrix >= edge_weight_threshold
    if verbose:
        print(f"original edge count: {adj_matrix.nonzero()[0].shape[0]:,}")
        print(
            f"edge count after pruning: {adj_matrix_pruned.nonzero()[0].shape[0]:,}"
        )
    return adj_matrix_pruned


def get_pairs_from_adj_matrix(adjacency_matrix: csr_matrix) -> np.ndarray:
    """ """
    return np.array(adjacency_matrix.nonzero()).T



def union_of_blocks(df: pd.DataFrame, block_dict: dict[str, list]) -> np.ndarray:
    """..."""

    adj_matrix_list = []
    for col_name, blocks in block_dict.items():
        pairs = get_pairs_from_blocks(blocks)
        adj_matrix_list.append(
            get_adjacency_matrix_from_pairs(pairs, (len(df), len(df)))
        )
    adj_matrix_union = get_union_of_adj_matrices(adj_matrix_list)
    new_pairs = get_pairs_from_adj_matrix(adj_matrix_union)
    return new_pairs


def get_union_of_adj_matrices(adj_matrix_list: list) -> csr_matrix:
    """ """
    adj_matrix = csr_matrix(adj_matrix_list[0].shape)
    for matrix in adj_matrix_list:
        adj_matrix += matrix
    return adj_matrix



def run_all_blocking_approaches(df: pd.DataFrame, cols: list) -> dict:
    """TODO"""
    result = {
        'sb_pairs': [],
        'tb_pairs': [],
        'sn_pairs': []
              }
    return result