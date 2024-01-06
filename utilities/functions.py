import itertools

def get_combinations(list_of_params):
   combination = [] # empty list 
   for r in range(1, len(list_of_params) + 1):
      # to generate combination
      combination.extend(itertools.combinations(list_of_params, r))
   return combination


# TODO create a function to find optimal values for hyper parameters (e.g. n_neighbors) by taking min max median sort approach
# i.e. test min max median, find optimal window hi or lo then repeat

def find_most_important_features(X_train, model): 

    # Selecting the top 3 features
    feature_importance = pd.Series(index=X_train.columns, data=model.coef_)
    top_features = feature_importance.abs().nlargest(3).index

   


def calculate_optimal_weights_ensemble(predictions, true_labels):
    # Create a vector of weights from 0 to 1 with a step size of 0.05
    weight_values = np.arange(0, 1.05, 0.05)

    # Initialize variables to store the optimal weights and RMSE
    optimal_weights = None
    min_rmse = float('inf')

    # Iterate through all combinations of weights
    for weight_lr in weight_values:
        weight_lasso = 1 - weight_lr

        # Calculate ensemble predictions using the current weights
        ensemble_predictions = weight_lr * predictions[0] + weight_lasso * predictions[1]

        # Calculate RMSE for the ensemble
        rmse = calculate_rmse(ensemble_predictions, true_labels)

        # Update optimal weights if the current combination results in a lower RMSE
        if rmse < min_rmse:
            min_rmse = rmse
            optimal_weights = [weight_lr, weight_lasso]

    return optimal_weights, min_rmse

# use for later
def optimize_function(starting_point, function_to_optimize, step_size=0.1, min_step=0.0001, max_iter=100):
    current_point = starting_point
    current_value = function_to_optimize(current_point)

    iteration = 0
    while iteration < max_iter and step_size > min_step:
        higher_point = current_point + step_size
        lower_point = current_point - step_size

        higher_value = function_to_optimize(higher_point)
        lower_value = function_to_optimize(lower_point)

        if lower_value < current_value:
            current_point = lower_point
            current_value = lower_value
        elif higher_value < current_value:
            current_point = higher_point
            current_value = higher_value
        else:
            step_size *= 0.5  # Reduce step size if neither direction is an improvement

        iteration += 1

    return current_point, current_value

