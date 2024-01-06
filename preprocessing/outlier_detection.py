import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM



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
