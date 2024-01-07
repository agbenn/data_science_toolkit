import pytest
from bse_dsdm.preprocessing.outlier_detection import *
import numpy as np

@pytest.mark.unittest
def test_remove_outliers():
    data = {'A': [1, 2, 3, 4, 5, 20, 75, np.nan],
        'B': [5, 8, 12, 15, 18, 25, 45, np.nan]}
    df = pd.DataFrame(data)
    
    assert remove_outliers(df, removal_type="std", std_threshold=1).isna().sum().sum() == 4
    assert remove_outliers(df, removal_type="iqr").isna().sum().sum() == 4
    assert remove_outliers(df, removal_type="iso_forest").isna().sum().sum() in [6,8]
    assert remove_outliers(df, removal_type="local_outlier", local_n_neighbors=2).isna().sum().sum() == 6
    assert remove_outliers(df, removal_type="min_covariance", cov_contamination=.3).isna().sum().sum() in [6,8]
    assert remove_outliers(df, removal_type="local_outlier").isna().sum().sum() == 6


test_remove_outliers()