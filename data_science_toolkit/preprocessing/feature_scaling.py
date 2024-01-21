from scipy import stats 
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


def optimal_scale_features(X, y, columns_to_scale, model_type='binary', accuracy_test='accuracy'):
    """
    Parameters:
    - X (DataFrame): Features.
    - y (Series): Target variable.
    - model_type (str): Type of model ('binary', 'multiclass', 'regression').
    - accuracy_test (str): Scoring metric for model evaluation (default is 'accuracy').

    Returns:
    - Tuple: Dictionary with optimal accuracy score and scaling method.

    """
    X_to_scale = X[columns_to_scale]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scaling_methods = ['No transformation', 'z_score', 'min_max_mean', 'min_max', 'iqr']
        min_accuracy = None
        best_scaling_method = None
        best_data_set = None

        for scaling_method in scaling_methods:
            X_scaled = X_to_scale.copy()
            if scaling_method != 'No transformation':
                X_scaled = scale_features(X_to_scale.copy(), X_to_scale.columns, scaling_method)

            model = LogisticRegression()  # You can customize the logistic regression model as needed

            try:
                accuracy = cross_val_score(model, X_scaled, y, cv=3, scoring=accuracy_test).mean()
                print(f"{scaling_method}: {accuracy}")
                
                if min_accuracy is None or accuracy > min_accuracy:
                    min_accuracy = accuracy
                    best_scaling_method = scaling_method
                    best_data_set = X_scaled
            except Exception as e:
                print(f'An exception occurred when getting the accuracy for {scaling_method}')
                print(str(e))

        X[columns_to_scale] = best_data_set

        return {'optimal_accuracy': min_accuracy, 'optimal_scaling_method': best_scaling_method, 'optimal_X_data_set':X}
    

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