# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:24:30 2020

@author: Mr. DUBEY
"""


import pandas as pd
import numpy as np  

import matplotlib.pyplot as plt   
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn import metrics
from sklearn.metrics import r2_score

url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Student data is Ready")

X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values
X= X.reshape(-1, 1)
y= y.reshape(-1, 1)
test_size = 5/25
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=0) 
print("split done")
linear = LinearRegression()  
linear.fit(X_train, y_train) 


# Saving model to disk
pickle.dump(linear, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2]]))