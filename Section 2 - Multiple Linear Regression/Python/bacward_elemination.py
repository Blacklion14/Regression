# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:15:56 2021

@author: Kush
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Impporting Dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Encoding Categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('encoder' , OneHotEncoder() , [3])] , remainder="passthrough")
x = np.array(ct.fit_transform(x))

#Spliting Data Into Training And Test Set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size = 0.2 , random_state = 0)

#Training the model on training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)

#Predicting the test results
y_pred = regressor.predict(x_test)

#Building backward elemination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

x_opt = x[: , [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS()
