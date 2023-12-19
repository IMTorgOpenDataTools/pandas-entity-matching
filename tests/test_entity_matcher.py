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
indices = [
    1723, 3653, 3866, 4808, 1681, 3000, 1463, 406, 3132, 4824, 
    11857, 12107, 8343, 9426, 2157, 4116, 2559, 8232, 9545, 5525, 
    6320, 3280, 8937, 3866, 4162, 10874, 4162, 10874, 8180, 11426, 
    8629, 10491
    ]


def test_get_matches_single_field():
    samp_df = pp_df.iloc[indices].reset_index()
    field_config = em.field_config
    field_config.clear()
    field_config['title'] = 'fuzzy'

    Matcher = em.EntityMatcher(field_config)
    samp_df['Proposed Matches'] = Matcher.get_matches(samp_df)
    matched = samp_df[['title','group']][samp_df['Proposed Matches']!=''].sort_values('group')
    print(matched)

    assert matched.shape == (6,2)


def test_get_matches_multiple_fields():
    #TODO:account for multiple fields, here, not just `title`
    samp_df = pp_df.iloc[indices].reset_index()
    field_config = em.field_config
    field_config.clear()
    field_config['title'] = 'fuzzy'
    field_config['artist'] = 'fuzzy'
    field_config['album'] = 'fuzzy'

    Matcher = em.EntityMatcher(field_config)
    samp_df['Proposed Matches'] = Matcher.get_matches(samp_df)
    matched = samp_df[['title','artist','album','group']][samp_df['Proposed Matches']!=''].sort_values('group')
    print(matched)

    assert True == False #matched.shape == (6,4)


def test_get_matches_with_blocking():
    assert True == True


def test_run_all_scoring_methods():
    assert True == True