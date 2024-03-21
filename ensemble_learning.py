#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 5 21:51:29 2024

@author: candilsiz
"""

from sklearn.model_selection import train_test_split
import numpy as np

class StackedHybrid:
    """
    inputs: model_1, model_1, X_1, X_2
    model_1 := weak learner
    model_2 := strong learner
    X_1 := Dataframe only includes time series features
    X_2 := Dataframe  includes all time series features
    """
    def __init__(self, model_1, model_2):
        self.model_1 = model_1  # Linear Regression
        self.model_2 = model_2  # XGBoost
        
    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_pred_1 = self.model_1.predict(X_1)

        X_2_augmented = X_2.copy()
        X_2_augmented['LR_pred'] = y_pred_1  # append as new feature / stacking here
        
        X_2_train, X_2_val, y_train, y_val = train_test_split(X_2_augmented, y, test_size=0.15, shuffle=False)
        self.model_2.fit(X_2_train, y_train,
                         eval_set=[(X_2_val, y_val)],
                         eval_metric="rmse",
                         verbose=True)
        
    def predict(self, X_1, X_2):
        y_pred_1 = self.model_1.predict(X_1)
        
        X_2_augmented = X_2.copy()
        X_2_augmented['LR_pred'] = y_pred_1
        
        X_2_train, X_2_test, y_train, y_test = train_test_split(X_2_augmented, y, test_size=0.15, shuffle=False)
        y_pred_2 = self.model_2.predict(X_2_test)
        
        return y_pred_2, y_test

# not finished
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1  # Linear Regression
        self.model_2 = model_2  # XGBoost

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_pred_1 = self.model_1.predict(X_1)
        
        y_res = np.absolute(y - y_pred_1)  # extract residuals / boosting here

        X_2_train, X_2_val, y_train, y_val = train_test_split(X_2, y_res, test_size=0.2, shuffle=False)
        self.model_2.fit(X_2_train, y_train,
                        eval_set=[(X_2_val, y_val)],
                        eval_metric="rmse",
                        verbose=True)
        
        
    def predict(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_pred_1 = self.model_1.predict(X_1)
        
        y_res = np.absolute(y - y_pred_1) 

        X_2_train, X_2_test, y_train, y_test = train_test_split(X_2, y_res, test_size=0.2, shuffle=False)
        y_pred_2 = self.model_2.predict(X_2_test)

        return y_pred_2, y_test
