# %%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from data_science_toolkit.preprocessing.outlier_detection import *
from data_science_toolkit.preprocessing.optimal_outlier_detection import *
from data_science_toolkit.preprocessing.feature_scaling import *
from data_science_toolkit.preprocessing.value_encoder import *
from data_science_toolkit.preprocessing.value_imputer import *
from data_science_toolkit.exploratory_analysis.binary_model_exploration import *
from data_science_toolkit.exploratory_analysis.missing_values import *
from data_science_toolkit.accuracy_testing.error_testing_methods import *
from data_science_toolkit.graphing.graphing import *

class DataLoader:
    def __init__(self):
        self.train_split_data = None
        self.test_split_data = None
        self.test_data_set = None
        self.train_data_set = None

    def load_train_data(self, file_path):
        data = pd.read_csv(file_path)
        self.train_data_set = data

    def load_test_data(self, file_path):
        data = pd.read_csv(file_path)
        self.test_data_set = data

    def split_train_data(self, test_size=0.2, random_state=42):
        self.train_split_data, self.test_split_data = train_test_split(self.train_data_set, test_size=test_size, random_state=random_state)


    
'''
    def get_preprocessing_options(): 
        if scaling_method == 'z_score':
                data[column] = standardize_with_z_score(data[column])
            elif scaling_method == 'min_max_mean':
                data[column] = min_max_scale_with_mean(data[column])
            elif scaling_method == 'min_max':
                data[column] = min_max_scale(data[column])    
            elif scaling_method == 'iqr':
                data[column] = robust_scaling_with_iqr(data[column])
        feature_scaling_options = {
            'method_name':'scale_features',
            'parameters':'data, columns_to_scale, scaling_method',
            'scaling_methods': ["z_score",
            "min_max_mean",
            "min_max",
            "iqr"]
        }
'''
   
