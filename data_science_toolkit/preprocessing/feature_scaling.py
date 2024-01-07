from scipy import stats 
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#TODO add a function to minimize or maximize the error term instead of always minimizing
def optimal_scale_features(X,y, accuracy_test='neg_mean_squared_error'):
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
        z_score = scale_features(X,y, scaling_method='z_score')
        min_max_mean = scale_features(X,y, removal_type='min_max_mean')
        min_max = scale_features(X,y, removal_type='min_max')
        iqr = scale_features(X,y, removal_type='iqr')
        
        model = LogisticRegression()
        min_accuracy = None
        min_data_set = None
        for results in [z_score,min_max_mean,min_max,iqr]:
            X_train = X.dropna()
            y_train = y.iloc[X_train.index]
            accuracy = None
            try:
                accuracy = cross_val_score(model, X_train, y_train, cv=3, scoring=accuracy_test)
                print(accuracy)
            except Exception as e:
                print('an exception occured when getting the accuracy')
                print(str(e))
            
        
        return min_results
    

def scale_features(data,columns_to_scale, scaling_method='z_score'):
    '''
    cumulative method for individual methods
    options: z_score, min_max_mean, min_max, iqr
    :param object column_of_data: a column or series of a dataframe
    '''
    for column in columns_to_scale: 
        try:    
            if scaling_method == 'z_score':
                data[column] = standardize_with_z_score(data[column])
            elif scaling_method == 'min_max_mean':
                data[column] = min_max_scale_with_mean(data[column])
            elif scaling_method == 'min_max':
                data[column] = min_max_scale(data[column])    
            elif scaling_method == 'iqr':
                data[column] = robust_scaling_with_iqr(data[column])
        except Exception as e: 
            print('an error occurred scaling columns: ' + str(column))
            print('error: ' + str(e))

    return data

def standardize_with_z_score(column_of_data): 
    '''
    method standardizes a column of data using a z-score method
    :param object column_of_data: a column or series of a dataframe
    '''
    # Rescale the variables using z-score standardization
    return (column_of_data - column_of_data.mean()) / column_of_data.std()

def min_max_scale_with_mean(column_of_data):
    '''
    method standardizes a column of data using a min max denominator scaling with the distance to the mean in the numerator
    :param object column_of_data: a column or series of a dataframe
    '''
    column_of_data = (column_of_data-column_of_data.mean())/(max(column_of_data)-min(column_of_data))
    return column_of_data

def min_max_scale(column_of_data):
    '''
    method standardizes a column of data using min max scaling
    :param object column_of_data: a column or series of a dataframe
    '''
    min_value = min(column_of_data)
    max_value = max(column_of_data)
    normalized_data = []

    for value in column_of_data:
        normalized_value = (value - min_value) / (max_value - min_value)
        normalized_data.append(normalized_value)

    return normalized_data

def robust_scaling_with_iqr(column_of_data):
    '''
    method standardizes a column of data using an interquartile range for the divisor
    :param object column_of_data: a column or series of a dataframe
    '''
    IQR1 = stats.iqr(column_of_data, interpolation = 'midpoint') 
    column_of_data = (column_of_data-column_of_data.median())/IQR1
    return column_of_data