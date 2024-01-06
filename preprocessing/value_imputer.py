
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer


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
