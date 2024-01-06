import numpy as np
import pandas as pd
from bse_dsdm.preprocessing.outlier_detection import *
import warnings


#TODO add a function to minimize or maximize the error term instead of always minimizing
def remove_optimal_outliers(X,y, accuracy_test='neg_mean_squared_error'):
    """

        Parameters:
        - X (DataFrame): Features.
        - y (Series): Target variable.
        - model_type (str): Type of model ('binary', 'multiclass', 'regression').
        - accuracy_test (str): Scoring metric for model evaluation (default is 'accuracy').

        Returns:
        - Tuple: DataFrame with outliers removed, dictionary with optimal accuracy score and parameter value.

        decrease iqr to increase outlier removal
        increase cov_contamination to increase outlier removal
        decrease std_threshold to increase outlier removal
        local_n_neighbors depends on the sample size
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iqr_results = remove_outliers(X,y, removal_type='iqr', accuracy_test=accuracy_test)
        iso_forest_results = remove_outliers(X,y, removal_type='iso_forest', accuracy_test=accuracy_test)
        min_cov_results = remove_outliers(X,y, removal_type='min_covariance', accuracy_test=accuracy_test)
        local_results = remove_outliers(X,y, removal_type='local_outlier', accuracy_test=accuracy_test)
        svm_results = remove_outliers(X,y, removal_type='svm', accuracy_test=accuracy_test)

        min_results = None
        for results in [iqr_results,iso_forest_results,min_cov_results,local_results,svm_results]:
            print(results)
            print(results['accuracy_score'].mean())
            if 'accuracy_score' in results.keys() and results['accuracy_score'] is not None and results['accuracy_score'].mean() < 0:
                if min_results is None or (results['accuracy_score'].mean() > min_results['accuracy_score'].mean()):
                    min_results = results
        
        return min_results
