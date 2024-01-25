from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN,\
    BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, \
    EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, \
    OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

'''
use a confusion matrix to figure out if resampling is needed i.e. the balance

Resampling treats unbalanced datasets. 
under-sampling: which entails removing samples from the majority class
over-sampling: which involves adding more examples from the minority class

RandomUnderSampler: randomly select a subset of data from the majority class
RandomOverSampler: randomly select samples from the minority class with replacement
SMOTE: Synthetic Minority Over-sampling Technique - create new samples from the minority class
    - does not make any distinction between easy and hard samples to classify
    - can create noisy samples by connecting inliers and outliers
ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning 
    - create new samples from the minority class with a higher concentration on hard to learn samples
    - uses a weighted distribution for different minority class samples according to their level of difficulty in learning
SMOTENC: SMOTE for continuous and categorical features
    - takes categorical features into account when oversampling
    - can be used with any of the SMOTE variants
SMOTEN: SMOTE for only categorical features
BorderlineSMOTE: SMOTE with borderline samples
SVMSMOTE: SMOTE with SVM samples
KMeansSMOTE: SMOTE with KMeans samples


Under-sampling methods:

RandomUnderSampler:
 - Use when you want to randomly reduce the majority class samples to balance the class distribution.
 - It randomly selects samples from the majority class without considering their proximity to other samples.

ClusterCentroids:
 - Use when you want to cluster the majority class samples and then select centroids from each cluster to balance the class distribution.
 - It creates clusters of majority class samples and selects centroids as representative samples.

NearMiss:
 - Use when you want to select samples from the majority class based on their proximity to the minority class samples.
 - It selects samples from the majority class that are closest to the minority class samples.

EditedNearestNeighbours:
 - Use when you want to remove samples from the majority class that are misclassified by their nearest neighbors.
 - It removes samples from the majority class that are misclassified by their nearest neighbors from the same class.

RepeatedEditedNearestNeighbours:
 - Use when you want to iteratively remove misclassified samples from the majority class using EditedNearestNeighbours.
 - It repeatedly applies EditedNearestNeighbours to remove misclassified samples until a stopping criterion is met.

AllKNN:
 - Use when you want to remove samples from the majority class that have the same label as the majority of their K nearest neighbors.
 - It removes samples from the majority class that have the same label as the majority of their K nearest neighbors.

CondensedNearestNeighbour:
 - Use when you want to select a subset of the majority class samples that can represent the entire majority class.
 - It starts with an empty set and adds samples from the majority class that are misclassified by their nearest neighbors from any class.

OneSidedSelection:
 - Use when you want to select samples from the majority class that are well separated from the minority class.
 - It selects samples from the majority class that are farthest from the decision boundary.

NeighbourhoodCleaningRule:
 - Use when you want to remove samples from the majority class that are misclassified by their nearest neighbors from any class.
 - It removes samples from the majority class that are misclassified by their nearest neighbors from any class.

InstanceHardnessThreshold:
 - Use when you want to estimate the hardness of each sample and remove the samples that are considered hard.
 - It estimates the hardness of each sample using a classifier and removes the samples that are above a certain hardness threshold.

'''


def SMOTE_over_resampling(X, y): 
    # Initialize SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # Fit and transform the data using SMOTE to perform over-sampling
    return  smote.fit_resample(X, y)

def ADASYN_over_resampling(X, y):
    # Initialize ADASYN (Adaptive Synthetic Sampling Approach for Imbalanced Learning)
    adasyn = ADASYN(sampling_strategy='auto', random_state=42)

    # Fit and transform the data using ADASYN to perform over-sampling
    return adasyn.fit_resample(X, y)

def under_sampling(X, y):
    # Initialize the RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

    # Fit and transform the data using random under-sampling
    return  rus.fit_resample(X, y)

def over_sampling(X, y):
    # Initialize the RandomOverSampler
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)

    # Fit and transform the data using random over-sampling
    return ros.fit_resample(X, y)
