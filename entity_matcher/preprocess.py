#!/usr/bin/env python3
"""
Preprocess
"""

__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


def preprocess_df_columns(df, cols=[]):
    """Process pandas DF columns"""
    for col in cols:
        df[col] = (
            df[col]
            .str.lower()
            .replace("[^a-z0-9]", " ", regex=True)  # replacing special characters with a space
            .replace(" +", " ", regex=True)         # removing consecutive spaces
            .str.strip()                            # removing leading and tailing spaces
        )

    df.loc[df.number.notna(), "number"] = (
    df[df.number.notna()]
    .number.replace("[^0-9]", "", regex=True)              # removing non-digits
    .apply(lambda x: str(int(x)) if len(x) > 0 else None)  # removing leading zeros
    )

    return df
