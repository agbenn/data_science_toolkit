import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from math import floor, ceil
from sklearn.svm import SVC


# Function to compute SVM for classification and plot results and buffers depending on number of samples
def plot_svm(N=10, ax=None, myCost=1E10, myRandomS=0, mySD=0.6, mySamples=200):
    X, y = make_blobs(n_samples=mySamples, centers=2,
                      random_state=myRandomS, cluster_std=mySD)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=myCost)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='rainbow')
    ax.set_xlim(floor(min(X[:, 0])), ceil(max(X[:, 0])))
    ax.set_ylim(floor(min(X[:, 1])), ceil(max(X[:, 1])))
    plot_svc_decision_function(model, ax)

def plot_3d_scatter(data, x_col,y_col,z_col,color_col, x_title=None, y_title=None, z_title=None):
    # Create a 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=data[x_col],
        y=data[y_col],
        z=data[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=data[color_col],
            colorscale='Viridis',
            opacity=0.8
        )
    ))

    if x_title is None: 
        x_title = x_col
    if y_title is None:
        y_title = y_col
    if z_title is None:
        z_title = z_col

    title = x_title + ', ' + y_title + ', ' + z_title

    # Set labels and title
    fig.update_layout(scene=dict(
        xaxis_title=x_title,
        yaxis_title=y_title,
        zaxis_title=z_title),
        title=title
    )

    # Display the plot
    fig.show()


def multiplot_bar(): 
    # Obtain coefficients
    coefs_001 = lasso_0_01.coef_
    coefs_01 = lasso_0_1.coef_
    coefs_05 = lasso_0_5.coef_

    # Create bar plots for coefficients
    labels = X.columns

    # Set common y-axis limits
    y_min = -4
    y_max = 4

    # Create subplots with shared y-axis
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

    # Linear Regression Coefficients
    axes[0].bar(labels, coefs_001, color='b', alpha=0.7)
    axes[0].set_title('Coefficients Alpha = 0.01')
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Coefficient')
    axes[0].set_ylim([y_min, y_max])

    # Lasso Regression Coefficients
    axes[1].bar(labels, coefs_01, color='g', alpha=0.7)
    axes[1].set_title('Coefficients Alpha = 0.1')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Coefficient')
    axes[1].set_ylim([y_min, y_max])

    # Ridge Regression Coefficients
    axes[2].bar(labels, coefs_05, color='r', alpha=0.7)
    axes[2].set_title('Coefficients Alpha = 0.5')
    axes[2].set_xlabel('Features')
    axes[2].set_ylabel('Coefficient')
    axes[2].set_ylim([y_min, y_max])

    plt.tight_layout()
    plt.show()


def plot_heat_map(df):
    df_corr = df.corr()

    # Plot correlations
    # Remove upper triangle
    fig, ax = plt.subplots(figsize=(14,8))
    ax = sns.heatmap(df_corr, annot = True)

def plot_distributions(data, columns):
    """
    Plot distributions of specified columns in a dataframe using Viridis color palette.

    Parameters:
    - data: Pandas DataFrame
    - columns: List of column names to plot
    """
    # Set the color palette to Viridis
    sns.set_palette("viridis")

    # Set up subplots
    num_plots = len(columns)
    num_cols = 2  # Number of columns in the subplot grid
    num_rows = (num_plots + 1) // 2  # Number of rows in the subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

    # Flatten the 2D array of subplots for easier indexing
    axes = axes.flatten()

    # Plot each column's distribution
    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(data[col], bins=20, kde=True, color='skyblue', edgecolor='black', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def plot_distributions_stacked(dataframes, columns, labels, colors=None):
    """
    Plot stacked distributions of specified columns in multiple dataframes.

    Parameters:
    - dataframes: List of Pandas DataFrames
    - columns: List of column names to plot
    - colors: List of colors for each dataframe's distributions
    """
    # Example usage:
    # Assuming you have three DataFrames called 'df1', 'df2', and 'df3' with columns 'column1', 'column2', ...
    # plot_distributions_stacked([df1, df2, df3], ['column1', 'column2', 'column3'])
    
    warnings.filterwarnings("ignore")
    # Set default colors if not provided
    if colors is None:
        colors = sns.color_palette("husl", n_colors=len(dataframes))

    # Set up subplots
    num_plots = len(columns)
    num_cols = 2  # Number of columns in the subplot grid
    num_rows = (num_plots + 1) // 2  # Number of rows in the subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

    # Flatten the 2D array of subplots for easier indexing
    axes = axes.flatten()

    # Plot distributions for each column in the multiple dataframes
    for i, col in enumerate(columns):
        ax = axes[i]
        for j, df in enumerate(dataframes):
            sns.histplot(df[col], bins=20, kde=True, color=colors[j], edgecolor='black', label=labels[j], ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_decision_tree(clf, feature_names, class_names):
    # Plot the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_any_cat_matrix(dat, y_var, x_var, width=10, height=10):
    """Pretty prints a categorical matrix of counts as a figure

    Args:
        dat:  A data frame, each row is an observation, and has more than one categorical feature
        y_var: Categorical variable name, should exist in dat
        x_var: Categorical variable name, should exist in dat

    Returns:
        Just plots the occurrence matrix.
    """
    
    aux = dat[[x_var, y_var]].groupby([x_var, y_var]).size()
    aux = pd.DataFrame(aux)
    aux.reset_index(level=0, inplace=True)
    aux.reset_index(level=0, inplace=True)
    aux

    counts = aux.pivot_table(index=y_var, columns=x_var, fill_value=0)
    counts.columns = counts.columns.droplevel(level=0)

    fig, ax = plt.subplots(figsize=(width, height))
    sns.set(font_scale=0.7)
    sns.heatmap(counts, annot=True, fmt='g', cmap='Blues', ax=ax)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.show()

def plot_point(x, y, neighbors, ax=None):
    """Plots sample observation, and some neighbors
    Used for K-NN
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='rainbow')
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.scatter(neighbors[:, 0], neighbors[:, 1], s=200, linewidth=1, edgecolors='black', facecolors='None')


def grid_search_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          table=True,
                          display_all_params=True):

    '''Display grid search results
    modified from https://www.kaggle.com/grfiv4/displaying-the-results-of-a-grid-search
    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    table              boolean: should a table be produced?
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          '''
    from matplotlib      import pyplot as plt
    from IPython.display import display
    import pandas as pd

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_


    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
    scores_df = scores_df[scores_df.columns.drop(list(scores_df.filter(regex='time')))] #drop time parameters

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    if table:
        display(scores_df \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
def plot_roc_curve(y, y_pred_probabilities, class_labels, column =1, plot = True):
    """Plots ROC AUC
    """
    fpr, tpr, _ = roc_curve(y == column, y_pred_probabilities[:,column],drop_intermediate = False)
    roc_auc = roc_auc_score(y_true=y, y_score=y_pred_probabilities[:,1])
    print ("AUC: ", roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """ Convenience function to plot results and buffers, extracted from Python Data Science Handbook
    Used for SVM notebook
    """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, edgecolors='black', facecolors='None');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)