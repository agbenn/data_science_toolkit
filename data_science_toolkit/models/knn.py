'''
explore data
label encode categorical variables
split data into train and test
feature scaling (use standard scaler and only fit scaler on training data)
fit model with compute_plot_neighbors
    observe class imbalance
fit model with grid search 
    start grid search with wide range of hyperparameters
    use grid search graphing function 'grid_search_table_plot' to narrow down hyperparameters
predict on test data
predict_proba on test data
    ** better for class imbalances
plot a confusion matrix
    plot_confusion_matrix
plot a roc curve
    plot_roc_curve

look for additional class imbalance

parameters to tune: 
    look at different distance computation metrics
        euclidean - default
        manhattan - distance between two points measured along axes at right angles (usually higher than euclidean)
        minkowski - generalization of euclidean and manhattan (when p=1, manhattan, when p=2, euclidean) where p is a parameter 
            just use one or the other
        hamming - for all the features that are categories, do pairwise comparison of the target point 
            and each observation, and add +1 to the distance whenever a category doesn't match. 
            For example, if one observation is {Hair:'blond', Sex:'man'} and the other is 
            {Hair:'blond', Sex:'woman'}, those observations will have a Hamming distance of 0+1=1
            ** if using one-hot encoding, this is the same as euclidean
        weighted_hamming - instead of 0 distance when classes match, one can have Freq_{class}, so that, 
            even when matching, we have some significant distance if that class is very frequent
            ** if using one-hot encoding, this is the same as euclidean
            ** usefull when there is class imbalance
        jaccard - distance between two sets (usually higher than euclidean)
            ** only works with binary features
    
    look at differen algorithms
        ball_tree - creates a tree structure that partitions the data into a set of balls
            starts with the data with the largest spread and then recursively partitions the data into smaller balls 
            until each ball contains a minimum number of points called leafs
            ** good for high dimensional data
        kd_tree - creates a tree structure that partitions the data into a set of hypercubes
            starts with the data with the largest spread and then recursively partitions the data into smaller hypercubes 
            until each hypercube contains a minimum number of points called leafs
            ** good for low dimensional data (hypercube is more efficient than ball)
        brute - brute force search - highly computationally expensive
            ** good for low dimensional data (irregularities fit balls better)

do you need a rounding function for predict_proba?


'''

## We load the relevant modules
from data_science_toolkit.graphing.graphing import plot_point
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier 


