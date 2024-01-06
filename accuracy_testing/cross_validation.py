from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, accuracy_score
import numpy as np
from sklearn.metrics import accuracy_score

def k_folds_cross_validation(model, X, y, no_k_folds=3):
    """
    * Pros: k-fold cross-validation offers a balance between computational efficiency and performance estimation accuracy. It partitions the data into k subsets (folds) and runs the model k times, using each fold as a test set once.
    * Cons: As k increases, the computational cost also increases since you need to train and evaluate the model k times. Smaller k values might lead to higher variance in the estimation of performance.
    """

    # Step 4: Define the scoring metrics
    scoring = {'r_squared': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error)}

    # Step 5: Perform k-fold cross-validation using cross_validate
    cv_results = cross_validate(model, X, y, cv=no_k_folds, scoring=scoring)

    cv_results

    # Step 6: Print the cross-validation results
    for fold in range(k):
        print(f"Fold {fold + 1}:")
        print(f"Validation samples: {cv_results['test_r_squared'][fold]}")
        print(f"  R-squared value: {cv_results['test_r_squared'][fold].mean()}")
        print(f"  Mean Squared Error (MSE): {cv_results['test_mse'][fold].mean()}\n")

    # Step 7: Calculate and print the final model accuracy (average R-squared and MSE)
    final_r_squared = cv_results['test_r_squared'].mean()
    final_mse = cv_results['test_mse'].mean()
    print("Final Average R-squared:", final_r_squared)
    print("Final Average MSE:", final_mse)


def stratified_cross_validation(model, X, y, no_k_folds):
    """
    Stratified cross-validation is particularly useful when dealing with imbalanced datasets, 
    where the class distribution is skewed, and accurate representation of all classes is necessary.
    """

    skf = StratifiedKFold(n_splits=no_k_folds, shuffle=True, random_state=42)

    # Step 5: Perform Stratified Cross-Validation and calculate accuracy for each fold
    accuracy_values = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}:")

        # Split the data into training and validation sets
        x1_train, x1_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the logistic regression model
        model.fit(x1_train, y_train)

        # Validate the model
        y_pred = model.predict(x1_val)

        # Calculate the accuracy for this fold
        accuracy = accuracy_score(y_val, y_pred)

        # Append the accuracy to the list
        accuracy_values.append(accuracy)

        print(f"  Accuracy: {accuracy:.4f}\n")

    # Step 6: Calculate and print the final accuracy (average accuracy)
    final_accuracy = np.mean(accuracy_values)
    print("Final Average Accuracy:", final_accuracy)


def leave_one_out_validation(model, X, y):
    """
    * Pros: LOOCV provides an almost unbiased estimate of model performance since it leaves out one data point as the test set in each iteration, leading to a large number of iterations.
    * Cons: LOOCV is computationally expensive, especially on large datasets, as it requires fitting the model N times, where N is the number of data points in the dataset. This can lead to significant time and resource requirements, making it impractical for large datasets.
    """
    # Step 2: Create the LeaveOneOut cross-validator
    loo = LeaveOneOut()

    # Step 3: Perform Leave-One-Out Cross-Validation using sklearn
    mse_values = []

    for train_index, val_index in loo.split(x):
        print(f"Validation for data point {val_index[0]+1}:")

        # Split the data into training and validation sets
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the linear regression model
        model.fit(x_train.reshape(-1, 1), y_train)

        # Validate the model
        y_pred = model.predict(x_val.reshape(1, -1))

        # Calculate the Mean Squared Error (MSE) for this fold
        mse = mean_squared_error([y_val], y_pred)
        mse_values.append(mse)

        print(f"  Training samples: {len(x_train)}")
        print(f"  Mean Squared Error (MSE): {mse}\n")

    # Step 4: Calculate and print the final model accuracy (average MSE)
    final_mse = np.mean(mse_values)
    print("Final Average MSE:", final_mse)


