"""
Test Utility Functions
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

def run_matcher():
    samp_df = pp_df.iloc[indices].reset_index()
    field_config = em.field_config
    field_config.clear()
    field_config['blocking'] = {}
    field_config['scoring'] = {}
    field_config['scoring']['title'] = 'fuzzy'

    Matcher = em.EntityMatcher(field_config)
    samp_df['Proposed Matches'] = Matcher.get_matches(samp_df)
    return samp_df

samp_df = run_matcher()



def test_review_get_similar_text():
    result_df = em.utils.get_similar_text(df=samp_df, 
                                          target_column='title', 
                                          text='the', 
                                          match_column='Proposed Matches'
                                          )
    assert result_df.shape[0] == 11


def test_summary_df_sparsity_summary():
    result_df = em.utils.df_sparsity_summary(df=samp_df,
                                             xcol='SourceID', 
                                             ycols=['title', 'language']
                                             )
    assert result_df.shape == (5,3)