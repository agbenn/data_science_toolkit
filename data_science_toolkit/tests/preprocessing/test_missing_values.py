import pandas as pd
from bse_dsdm.preprocessing.missing_values import *
import pytest

@pytest.mark.unittest
def test_remove_columns_with_na():
    data = {
        'A': [1, None, None, 4, 5],
        'B': [None, 2, 3, 4, None],
        'C': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)

    # If the threshold = 30%, column 'A' and 'B' should be dropped
    threshold_30 = 30
    df_result_30 = remove_columns_with_na(df, threshold_30)
    assert 'A' not in df_result_30.columns
    assert 'B' not in df_result_30.columns
    assert 'C' in df_result_30.columns

@pytest.mark.unittest
def test_get_columns_by_type():
    
    data = {
        'A': ['a', 'b', 'c', None, 'e'],
        'B': [1, 2, 3, 4, None],
        'C': [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)

    # Test the function
    categorical_na_cols, numerical_na_cols = get_na_columns_by_type(df)
    

    assert 'A' in categorical_na_cols
    assert 'B' in numerical_na_cols
    
    assert len(categorical_na_cols) == 1  
    assert len(numerical_na_cols) == 1  


