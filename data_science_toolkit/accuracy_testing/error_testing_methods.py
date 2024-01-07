from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, r2_score, confusion_matrix, f1_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix
import numpy as np
def evaluate_classification_metrics(y_true, y_pred):
    """
    Evaluate various classification metrics and print a summary based on the type of model.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels

    Returns:
    None
    """

    if len(np.unique(y_true.values)) == 2:  # Binary Classification Metrics
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred)

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Precision
        precision = precision_score(y_true, y_pred)

        # Recall
        recall = recall_score(y_true, y_pred)

        # R-squared
        r_squared = r2_score(y_true, y_pred)

        # Other Accuracy Statistics
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)


        # Print Summary
        print("------ Binary Classification Metrics Summary ------")
        print(f"ROC-AUC Score: {roc_auc:.4f} - AUC represents the area under the Receiver Operating Characteristic curve.")
        print(f"Accuracy: {accuracy:.4f} - The proportion of correctly classified instances.")
        print(f"Precision: {precision:.4f} - The ability of the classifier not to label as positive a sample that is negative.")
        print(f"Recall: {recall:.4f} - The ability of the classifier to find all the positive samples.")
        print(f"R-squared: {r_squared:.4f} - The proportion of the variance in the dependent variable that is predictable from the independent variable.")
        print(f"F1 Score: {f1:.4f} - The harmonic mean of precision and recall.")
        print(f"Matthews Correlation Coefficient: {mcc:.4f} - Measures the quality of binary classifications.")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f} - The arithmetic mean of sensitivity and specificity.")
       

    elif len(np.unique(y_true.values)) > 2:  # Multiclass Classification Metrics
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 Score, Support
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')


        # Print Summary
        print("------ Multiclass Classification Metrics Summary ------")
        print(f"Accuracy: {accuracy:.4f} - The proportion of correctly classified instances.")
        print(f"Precision: {precision:.4f} - Weighted average precision across all classes.")
        print(f"Recall: {recall:.4f} - Weighted average recall across all classes.")
        print(f"F1 Score: {f1:.4f} - Weighted average F1 score across all classes.")
       

    else:
        print("Invalid number of unique classes in the true labels.")