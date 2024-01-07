from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestModel:
    def __init__(self, feature_columns, target_column, train_data, test_data, params=None):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.train_data = train_data
        self.test_data = test_data
        self.params = params

        if self.params:
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        X_train = self.train_data[self.feature_columns]
        y_train = self.train_data[self.target_column]
        self.model.fit(X_train, y_train)

    def predict(self):
        X_test = self.test_data[self.feature_columns]
        y_test = self.test_data[self.target_column]

        predictions = self.model.predict(X_test)
        return predictions

    def get_accuracy(self, predictions, y_test):
        accuracy = accuracy_score(y_test, predictions)
        return accuracy




