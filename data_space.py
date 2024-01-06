# %%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class MLDataLoader:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.data = None

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        self.data = data

    def split_data(self, test_size=0.2, random_state=42):
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)

    def get_train_data(self):
        if self.train_data is None:
            raise ValueError("Train data not loaded.")
        return self.train_data

    def get_test_data(self):
        if self.test_data is None:
            raise ValueError("Test data not loaded")
        return self.test_data
   
