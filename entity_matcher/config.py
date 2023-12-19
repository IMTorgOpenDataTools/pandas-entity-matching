#!/usr/bin/env python3
"""
Configuration
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"



field_config = {
    """Config for EntityMatcher

    The following is the minimum useable structure:
        >>> field_config.clear()
        >>> field_config['blocking'] = {}
        >>> field_config['scoring']['COLUMN_NAME'] = 'SCORE_METHOD'
    
    """
    "blocking":{
        "operation": "standard",
        "process": "purge"
    },
    "scoring":{
        # <field>: <sim_type>
        "title": "fuzzy",
        "artist": "fuzzy",
        "album": "fuzzy",
        "number": "exact",
    }
}


def validate_config(field_config: dict) -> bool:
    """Validate the shape of the `field_config`
    """
    assert all([item in list(field_config.keys()) for item in ['blocking', 'scoring']])

    assert (
        field_config['blocking'] == {} or 
        list(field_config['blocking'].keys()) == ['operation', 'process']
    )
    for op_or_proc in field_config['blocking'].keys():
        assert type(field_config['blocking'][op_or_proc]) == str

    assert list(field_config['scoring'].keys()).__len__() > 0
    for field in field_config['scoring'].keys():
        assert type(field_config['scoring'][field]) == str
    
    return True