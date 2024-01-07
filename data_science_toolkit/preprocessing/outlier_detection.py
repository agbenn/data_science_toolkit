import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
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
        iqr_results = remove_outliers_grid_search(X,y, removal_type='iqr', accuracy_test=accuracy_test)
        iso_forest_results = remove_outliers_grid_search(X,y, removal_type='iso_forest', accuracy_test=accuracy_test)
        min_cov_results = remove_outliers_grid_search(X,y, removal_type='min_covariance', accuracy_test=accuracy_test)
        local_results = remove_outliers_grid_search(X,y, removal_type='local_outlier', accuracy_test=accuracy_test)
        svm_results = remove_outliers_grid_search(X,y, removal_type='svm', accuracy_test=accuracy_test)

        min_results = None
        for results in [iqr_results,iso_forest_results,min_cov_results,local_results,svm_results]:
            print(results)
            print(results['accuracy_score'].mean())
            if 'accuracy_score' in results.keys() and results['accuracy_score'] is not None and results['accuracy_score'].mean() < 0:
                if min_results is None or (results['accuracy_score'].mean() > min_results['accuracy_score'].mean()):
                    min_results = results
        
        return min_results

def remove_outliers_grid_search(X, y, accuracy_test='neg_mean_squared_error', removal_type="std", param_grid=None):
    """
    Parameters:
    - X (DataFrame): Features.
    - y (Series): Target variable.
    - model_type (str): Type of model ('binary', 'multiclass', 'regression').
    - accuracy_test (str): Scoring metric for model evaluation (default is 'accuracy').
    - removal_type (str): Type of outlier removal method ('std', 'iqr', 'iso_forest', 'min_covariance', 'local_outlier', 'svm').
    - param_grid (dict): Dictionary specifying the range of parameter values for each removal method (default is None).
    Returns:
    - Tuple: DataFrame with outliers removed, dictionary with optimal accuracy score and parameter value.
    decrease iqr to increase outlier removal
    increase cov_contamination to increase outlier removal
    decrease std_threshold to increase outlier removal
    local_n_neighbors depends on the sample size
    """

    model = LogisticRegression()

    print('removing outliers with ' + removal_type + ' method')

    if param_grid == None: 
        param_grid = {
            'std':np.arange(1,3,.5),
            'iqr':np.arange(1,3,.5),
            'iso_forest':np.arange(0.1,.5,.04),
            'min_covariance':np.arange(0.1,.5,.04),
            "local_outlier":np.arange(1,100,5),
            "svm":np.arange(0.1,.5,.04)
        }
    elif removal_type not in param_grid.keys():
        print('invalid removal type')

    min_accuracy = None
    best_param = None
    best_df = None
    for param_val in param_grid[removal_type]:
        try:
            print('param_val: ' + str(param_val))
            if removal_type == "std":
                X = remove_outliers_std_deviation(X, param_val)
            elif removal_type == "iqr":
                X = remove_outliers_iqr(X, param_val)
            elif removal_type == "iso_forest":
                X = remove_outliers_iso_forest(X, "auto")
            elif removal_type == "min_covariance":
                X = remove_outliers_min_covariance_det(X, param_val)
            elif removal_type == "local_outlier":
                X = remove_outliers_local_outlier(X, param_val)
            elif removal_type == "svm":
                X = remove_outliers_one_class_svm(X, param_val)
            else:
                print('an unknown error occurred. probably something with the specified removal type.')

            X_train = X.dropna()
            y_train = y.iloc[X_train.index]
            print(X_train)
            print(y_train)
            accuracy = None
            try:
                accuracy = cross_val_score(model, X_train, y_train, cv=3, scoring=accuracy_test)
                print(accuracy)
            except Exception as e:
                print('an exception occured when getting the accuracy')
                print(str(e))

            if min_accuracy is None or (accuracy is not None and accuracy.mean() < 0 and accuracy.mean() > min_accuracy.mean()):
                min_accuracy = accuracy
                best_param = param_val
                best_df = X.copy()
        except ValueError as ve:
            print('the outlier is taking too many values off. Function returning. ' + str(ve))
            print(min_accuracy)
            print(best_param)
            return {'removal_type':removal_type, 'data':best_df, 'accuracy_score':min_accuracy, 'optimal_param_value':best_param}

    print(min_accuracy)
    print(best_param)
    return {'removal_type':removal_type, 'data':best_df, 'accuracy_score':min_accuracy, 'optimal_param_value':best_param}


def remove_outliers(df, removal_type="std", cov_contamination=0.3, std_threshold=3, iqr_multiplier=1.5, local_n_neighbors=2):
    """
        decrease iqr to increase outlier removal
        increase cov_contamination to increase outlier removal
        decrease std_threshold to increase outlier removal
        local_n_neighbors depends on the sample size
    """
    new_data = {}

    if removal_type == "std":
        return remove_outliers_std_deviation(df, std_threshold)
    elif removal_type == "iqr":
        return remove_outliers_iqr(df, iqr_multiplier)
    elif removal_type == "iso_forest":
        return remove_outliers_iso_forest(df, "auto")
    elif removal_type == "min_covariance":
        return remove_outliers_min_covariance_det(df, cov_contamination)
    elif removal_type == "local_outlier":
        return remove_outliers_local_outlier(df,local_n_neighbors)
    elif removal_type == "svm":
        return remove_outliers_one_class_svm(df)
    else:
        print("no method selected")


def remove_outliers_iqr(data, iqr_multiplier=1.5): 
    """
    Remove outliers outside of the interquartile range (IQR) in specified columns of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to consider for outlier removal
    - multiplier: multiplier for the IQR to determine the outlier threshold

    Returns:
    - DataFrame without outliers in the specified columns
    """

    data_no_outliers = data.copy()

    for column in data.columns:
        # Calculate the IQR for the column
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the upper and lower bounds for outliers
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        # Replace outliers outside of the bounds with NaN
        data_no_outliers[column] = np.where((data_no_outliers[column] < lower_bound) | (data_no_outliers[column] > upper_bound), np.nan, data_no_outliers[column])

    return data_no_outliers


def remove_outliers_std_deviation(data, threshold=3):
    '''
    method removes outliers using a standard deviation
    :param object column_of_data: a column or series of a dataframe
    :param int threshold: level at which outliers are trimmed by std dev
    '''

    data_no_outliers = data.copy()

    for col in data.columns:
        mean_value = data[col].mean()
        std_value = data[col].std()

        lower_bound = mean_value - (std_value * threshold)
        upper_bound = mean_value + (std_value * threshold)

        # Replace outliers outside of the bounds with NaN
        data_no_outliers[col] = np.where((data_no_outliers[col] < lower_bound) | (data_no_outliers[col] > upper_bound), np.nan, data_no_outliers[col])

    return data_no_outliers


def remove_outliers_iso_forest(data, contamination=0.01):
    '''
    Removing outliers based on the Isolation Forest algorithm
    :param DataFrame data: a DataFrame
    :param float contamination: level at which outliers are trimmed (0-0.5)
    :return: DataFrame with outliers set to NaN
    '''
    nan_mask = data.isna()
    data = data.fillna(data.mean())
    iso_forest = IsolationForest(contamination=contamination)
    yhat = iso_forest.fit_predict(data)
    # Set imputed outliers to NaN, keeping original NaN values
    data[yhat == -1] = np.nan
    data = data.where(~nan_mask, np.nan)
    return data


def remove_outliers_elliptic_envelope(data, contamination=0.01):
    '''
    Removing outliers based on the Elliptic Envelope algorithm
    :param DataFrame data: a DataFrame
    :param float contamination: level at which outliers are trimmed (0-0.5)
    :return: DataFrame with outliers set to NaN
    '''
    nan_mask = data.isna()
    data = data.fillna(data.mean())
    ee = EllipticEnvelope(contamination=contamination)
    yhat = ee.fit_predict(data)
    # Set imputed outliers to NaN, keeping original NaN values
    data[yhat == -1] = np.nan
    data = data.where(~nan_mask, np.nan)
    return data

def remove_outliers_local_outlier(data, n_neighbors=2):
    '''
    Removing outliers based on the Local Outlier Factor algorithm
    :param DataFrame data: a DataFrame
    :param int n_neighbors: number of neighbors to consider
    :return: DataFrame with outliers set to NaN
    '''
    nan_mask = data.isna()
    data = data.fillna(data.mean())
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    yhat = lof.fit_predict(data)
    # Set imputed outliers to NaN, keeping original NaN values
    data[yhat == -1] = np.nan
    data = data.where(~nan_mask, np.nan)
    return data

def remove_outliers_one_class_svm(data):
    '''
    Removing outliers based on the One-Class SVM algorithm
    :param DataFrame data: a DataFrame
    :return: DataFrame with outliers set to NaN
    '''
    nan_mask = data.isna()
    data = data.fillna(data.mean())
    svm_model = OneClassSVM(nu=0.01)
    yhat = svm_model.fit_predict(data)
    # Set imputed outliers to NaN, keeping original NaN values
    data[yhat == -1] = np.nan
    data = data.where(~nan_mask, np.nan)
    return data

def remove_outliers_min_covariance_det(data, contamination=0.01):
    '''
    Removing outliers based on a Gaussian distribution for each column
    :param DataFrame data: a DataFrame
    :param float contamination: level at which outliers are trimmed (0-0.5)
    :return: DataFrame with outliers removed for each column
    '''
    nan_mask = data.isna()
    data = data.fillna(data.mean())
    ee = EllipticEnvelope(contamination=contamination)
    yhat = ee.fit_predict(data)
    # Set imputed outliers to NaN, keeping original NaN values
    data[yhat == -1] = np.nan
    data = data.where(~nan_mask, np.nan)
    return data
