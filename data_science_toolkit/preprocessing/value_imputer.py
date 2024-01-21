
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_iterative_imputer
from joblib import Parallel, delayed
import warnings

from sklearn.impute import KNNImputer, IterativeImputer, MissingIndicator


def optimal_impute_method_parallel(X, y, columns_to_impute, impute_methods=['constant', 'ffill', 'bfill', 'mean', 'mode', 'knn', 'binary', 'iterative'],
                                   n_neighbors=2, accuracy_test='accuracy', n_jobs=-1):
    X_copy = X[columns_to_impute].copy()
    results = Parallel(n_jobs=n_jobs)(
        delayed(impute_single_method)(X_copy, y, columns_to_impute, method, n_neighbors=n_neighbors, accuracy_test=accuracy_test)
        for method in impute_methods
    )

    # Filter out unsuccessful imputation methods
    results = [result for result in results if result[1] is not None]

    # Find the best result based on accuracy
    best_result = max(results, key=lambda x: x[1]) if results else None

    X[columns_to_impute] = best_result[2]
    return {'optimal_accuracy': best_result[1] if best_result else None,
            'optimal_impute_method': best_result[0] if best_result else None,
            'optimal_imputed_df': X if best_result else None}

def impute_single_method(X, y, columns_to_impute, impute_method, n_neighbors=2, accuracy_test='accuracy'):
    X_copy = X[columns_to_impute]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        imputed_df = impute_values(X_copy.copy(), impute_type=impute_method, n_neighbors=n_neighbors)
        print(impute_method)
        X_imputed = imputed_df.dropna()
        y_imputed = y.iloc[X_imputed.index]

        model = LogisticRegression()

        try:
            accuracy = cross_val_score(model, X_imputed, y_imputed, cv=3, scoring=accuracy_test).mean()
            print(f"{impute_method}: {accuracy}")
            X[columns_to_impute] = imputed_df
            return impute_method, accuracy, X

        except Exception as e:
            print(f'An exception occurred when getting the accuracy for {impute_method}')
            print(str(e))
            return impute_method, None, None

def impute_values(df, impute_type='constant', impute_constant=None, n_neighbors=2):
    """
    allows for value impution
    options: ffill, bfill, mean, knn, mode
    """
    filled_df = None

    if impute_type == 'constant' and impute_constant is not None:
        filled_df = df.fillna(value=impute_constant)
    elif impute_type == 'constant' and impute_constant is None:
        print('must input a impute value constant')
        return
    elif impute_type == 'ffill':
        print(filled_df)
        filled_df = df.ffill()
        print(filled_df)
    elif impute_type == 'bfill':
        filled_df = df.bfill()
    elif impute_type == 'mean':
        filled_df = df.fillna(df.mean(numeric_only=True))
    elif impute_type == 'mode':
        filled_df = df.fillna(df.mode())
    elif impute_type == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # Perform KNN imputation
        imputed_data = imputer.fit_transform(df)

        # Convert the imputed data back to DataFrame
        filled_df = pd.DataFrame(imputed_data, columns=df.columns)
    elif impute_type == 'binary':

        indicator = MissingIndicator()
        # Fit and transform the data
        missing_indicator = indicator.fit_transform(df)

        # Convert the result back to a DataFrame
        filled_df = pd.DataFrame(missing_indicator, columns=[col + '_missing' for col in df.columns])

    elif impute_type == 'iterative':
        # A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.
        # Instantiate the IterativeImputer
        imputer = IterativeImputer(max_iter=10, random_state=42)

        # Fit and transform the data
        imputed_data = imputer.fit_transform(df)

        # Convert the result back to a DataFrame
        filled_df = pd.DataFrame(imputed_data, columns=df.columns)

    else:
        print('invalid method selection')

    return filled_df


def impute_categorical(df, columns_to_impute):
    filled_df = df.copy()

    for col in columns_to_impute:

        mode_val = df[col].mode().iloc[0]
        filled_df[col].fillna(mode_val, inplace=True)
    print(filled_df)
    return filled_df
