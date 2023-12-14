"""
Test Entity Matcher
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


import entity_matcher as em

from tests.setup import prepare_dataset


#setup
pp_df = prepare_dataset(preprocess=True)


def test_get_matches_basic():
    indices = [
        1723, 3653, 3866, 4808, 1681, 3000, 1463, 406, 3132, 4824, 
        11857, 12107, 8343, 9426, 2157, 4116, 2559, 8232, 9545, 5525, 
        6320, 3280, 8937, 3866, 4162, 10874, 4162, 10874, 8180, 11426, 
        8629, 10491
        ]
    samp_df = pp_df.iloc[indices].reset_index()
    field_config = em.field_config
    field_config.clear()
    field_config['title'] = 'fuzzy'

    Matcher = em.EntityMatcher(samp_df, field_config)
    samp_df['Proposed Matches'] = Matcher.get_matches()
    matched = samp_df[['title','group']][samp_df['Proposed Matches']!=''].sort_values('group')
    print(matched)

    assert matched.shape == (6,2)