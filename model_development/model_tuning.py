import itertools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def get_combinations(list_of_params):
   combination = [] # empty list 
   for r in range(1, len(list_of_params) + 1):
      # to generate combination
      combination.extend(itertools.combinations(list_of_params, r))
   return combination

#TODO find other error minimization techniques
def find_optimal_model_params_minimizing(model, possible_features, X, y, min_num_of_features=2):

    param_combinations = [i for i in get_combinations(possible_features) if len(i) > min_num_of_features]

    best_mse = None
    best_params = None

    #find optimal parameters
    for combo in param_combinations:

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)
        
        model.fit(X_train, y_train)  # perform linear regression
        
        y_pred = model.predict(X_test)  # make predictions

        mse = mean_squared_error(y_test, y_pred)

        if best_mse == None: 
            best_mse = mse
            best_params = combo
        elif mse < best_mse: 
            best_mse = mse
            best_params = combo
    
    return best_mse, best_params


def perform_grid_search(model, param_grid):
    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=model.model, param_grid=param_grid, cv=5,
                               scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(model.train_data[model.feature_columns], model.train_data[model.target_column])

    # Print the best hyperparameters and corresponding accuracy
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)

    # Evaluate the model with the best hyperparameters on the test set
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(model.test_data[model.feature_columns])
    accuracy = model.get_accuracy(predictions, model.test_data[model.target_column])
    print("Test Accuracy with Best Hyperparameters:", accuracy)

    return best_model
