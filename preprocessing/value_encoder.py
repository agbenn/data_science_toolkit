import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def encode_categorical_columns(df, column=None):
    """
    nominal or categorical i.e. blue red green
    """
    column_to_encode = df[[column]]
    # Apply one-hot encoding
    encoder = OneHotEncoder(drop='first', sparse=False)

    encoded_data = encoder.fit_transform(column_to_encode)
    # Convert the encoded data back to a dataframe

    encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column]))
    df = pd.concat([df, encoded_data], axis=1)
    return df


def encode_ordinal_columns(data, columns=None, mapping=None):
    """
    levels of category or ordinal i.e. low medium high
    """
    if mapping:
        for columns, mapping in mapping.items():
            data[columns] = data[columns].map(mapping)
    else:
        # Apply label encoding
        encoder = LabelEncoder()
        for col in columns:
            encoded_data = encoder.fit_transform(data[col])
            full_col_name = col + '_encoded'
            data[full_col_name] = encoded_data

    return data


def target_encode_column(df, categorical_groupby_value, compute_column, compute_type='mean'):
    """ 
    useful when there is a correlation between the categorical variable and the other variable.
    i.e. mean GDP grouped by country

    """

    grouped_by_mapping = None

    # Calculating the
    if compute_type == 'mean':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].mean()
    elif compute_type == 'sum':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].sum()
    elif compute_type == 'count':
        grouped_by_mapping = df.groupby(categorical_groupby_value)[compute_column].count()

    # Encoding the categorical variable 'group_by_value' using target encoding
    return df[categorical_groupby_value].map(grouped_by_mapping)

def convert_to_age(df, column_name, reference_year):

    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    df[column_name] = reference_year - df[column_name].dt.year
    df.rename(columns={column_name: 'age'}, inplace=True)
    return df


def extract_string(df, column_name):

    df[column_name] = df[column_name].str.replace(', ', ',')
    df[column_name] = df[column_name].str.replace(' ', '_')
    df_encoded = df[column_name].str.get_dummies(sep=',')
    df = pd.concat([df, df_encoded], axis=1)
    return df