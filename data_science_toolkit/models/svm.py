'''
Models
For classification
Use the SVC function from the sklearn library for classification.

Main parameters are:

C: Penalty cost parameter for slack points, useful for preventing overfitting.

kernel: Kernel inner product function. It can be 'linear', 'rbf' (radial), 'poly', or 'sigmoid'. You can also try your own functions following the documentation.
probability: Set to True to enable probability estimates.
    - linear: no probability estimates
    - rbf: probability estimates - most common
    - poly: probability estimates - non-linear kernel for polynomial functions
    - sigmoid: probability estimates - similar to logistic regression
class_weight: Set to 'balanced' if you want automatic balancing of classes. 
random_state: Seed to control shuffling of data for probability estimates.
gamma: To be tuned for 'rbf', 'poly', and 'sigmoid'.
degree: Degree of the polynomial kernel for 'poly'.
coef0: Independent term to be tuned for 'poly' and 'sigmoid'.

As an alternative, you can explore the LinearSVC function, which provides extra features specifically for linear hyperplane classification.

For regression
Use the SVR function from the sklearn library for regression.

The main parameter that differs from SVC is:

epsilon: Similar to the margin concept in classification, it specifies an epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value. Penalties start beyond this epsilon tube.
'''