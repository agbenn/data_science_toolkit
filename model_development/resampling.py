from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

'''
use a confusion matrix to figure out if resampling is needed i.e. the balance

Resampling treats unbalanced datasets. 
under-sampling: which entails removing samples from the majority class
over-sampling: which involves adding more examples from the minority class

'''


def SMOTE_over_resampling(X, y): 
    # Initialize SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # Fit and transform the data using SMOTE to perform over-sampling
    return  smote.fit_resample(X, y)

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
