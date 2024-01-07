import pytest
import pandas as pd
import numpy as np
from bse_dsdm.preprocessing.value_imputer import *


@pytest.mark.unittest
def test_value_imputer():
    data = {'A': [1, 2, 3, None, 5, None, 75],
            'B': [5, 8, None, 15, 18, None, 45]}
    df = pd.DataFrame(data)

    assert impute_values(df, 'constant', impute_constant=0).isna().sum().sum() == 0
    assert impute_values(df, 'ffill').isna().sum().sum() == 0
    assert impute_values(df, 'bfill').isna().sum().sum() == 0
    assert impute_values(df, 'mean').isna().sum().sum() == 0
    assert impute_values(df, 'knn', n_neighbors=3).isna().sum().sum() == 0
    assert impute_values(df, 'constant', impute_constant=0).isna().sum().sum() == 0


@pytest.mark.unittest
def test_impute_categorical():
    data2 = pd.DataFrame({'years_of_contract': [1, 2, np.nan, 2],
                          'traits': ['defender', 'attacker', np.nan, 'defender']})

    # Specify columns to impute
    columns_to_impute = ['years_of_contract', 'traits']
    filled_df = impute_categorical(data2, columns_to_impute)

    # Check that missing values are filled
    assert filled_df['years_of_contract'].isnull().sum() == 0
    assert filled_df['traits'].isnull().sum() == 0

    # Check that missing values are actually filled with the most common value
    assert filled_df['years_of_contract'].tolist() == [1, 2, 2, 2]
    assert filled_df['traits'].tolist() == ['defender', 'attacker', 'defender', 'defender']

   # Check for shape
    assert filled_df.shape == data2.shape


test_value_imputer()