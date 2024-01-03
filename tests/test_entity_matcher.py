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


def test_get_matches_single_field_no_blocking():
    samp_df = pp_df.iloc[indices].reset_index()
    field_config = em.field_config
    field_config.clear()
    field_config['blocking'] = {}
    field_config['scoring'] = {}
    field_config['scoring']['title'] = 'fuzzy'

    Matcher = em.EntityMatcher(field_config)
    samp_df['Proposed Matches'] = Matcher.get_matches(samp_df)
    matched = samp_df[['title','Proposed Matches']][samp_df['Proposed Matches']!=''].sort_values('Proposed Matches')
    print(matched)

    assert matched.shape == (6,2)


def test_get_matches_single_field_with_blocking():
    samp_df = pp_df.iloc[indices].reset_index()
    field_config = em.field_config
    field_config.clear()
    field_config['blocking'] = {"operation": "standard", "process": "purge"} 
    field_config['scoring'] = {}
    field_config['scoring']['title'] = 'fuzzy'

    Matcher = em.EntityMatcher(field_config)
    samp_df['Proposed Matches'] = Matcher.get_matches(samp_df)
    matched = samp_df[['title','Proposed Matches']][samp_df['Proposed Matches']!=''].sort_values('Proposed Matches')
    print(matched)

    assert matched.shape == (6,2)


def test_get_matches_multiple_fields_with_blocking():
    samp_df = pp_df.iloc[indices].reset_index()
    field_config = em.field_config
    field_config.clear()
    field_config['blocking'] = {"operation": "standard", "process": "purge"} 
    field_config['scoring'] = {}
    field_config['scoring']['title'] = 'fuzzy'
    field_config['scoring']['artist'] = 'fuzzy'
    field_config['scoring']['album'] = 'fuzzy'
    #TODO:field_config['scoring']['number'] = 'exact'

    Matcher = em.EntityMatcher(field_config)
    samp_df['Proposed Matches'] = Matcher.get_matches(samp_df)
    matched = samp_df[['title','artist','album','Proposed Matches']][samp_df['Proposed Matches']!=''].sort_values('Proposed Matches')

    assert matched.shape == (4,4)

#NOTE: this test requires a long runtime
def test_get_matches_multiple_fields_with_blocking_ALL_DATA():
    raw = prepare_dataset(preprocess=False)
    raw_df = raw.reset_index()
    field_config = em.field_config
    field_config.clear()
    field_config['blocking'] = {"operation": "standard", "process": "purge"} 
    field_config['scoring'] = {}
    field_config['scoring']['title'] = 'fuzzy'
    field_config['scoring']['artist'] = 'fuzzy'
    field_config['scoring']['album'] = 'fuzzy'

    Matcher = em.EntityMatcher(field_config)
    samp_df = Matcher.preprocess(raw_df)
    samp_df['Proposed Matches'] = Matcher.get_matches(samp_df)
    matched = samp_df[['title','artist','album','Proposed Matches']][samp_df['Proposed Matches']!=''].sort_values('Proposed Matches')
    counts = matched['Proposed Matches'].value_counts()

    assert matched.shape == (96062, 4)
    assert list(counts[:3]) == [449, 290, 267]


def test_get_matches_with_combined_blocking():
    #combined_blocking refers to output, such as`token_blocking()``
    assert True == True


def test_get_matches_with_all_blocking_methods():
    assert True == True


def test_get_matches_with_all_blocking_all_scoring_methods():
    assert True == True


def test_run_all_scoring_methods():
    assert True == True