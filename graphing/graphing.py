import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix


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


