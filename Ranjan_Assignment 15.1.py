# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 01:37:32 2018

@author: Ranjan
"""
#pre-defined code to import all required packages and dataset
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
boston = load_boston()
#Converting data fields into DataFrame
bos = pd.DataFrame(boston.data)

#assigining coloumn names to DataFrame
bos.columns=boston.feature_names

#Copying existing DataFrame before adding Price col
XData=bos

#adding "Target" col as Price col
bos["Price"]=boston.target

#initializing variable for Linear Regression model
linreg=linear_model.LinearRegression()

#creating sp;its of train_test
X_train,X_test,y_train,y_test=train_test_split(XData,bos.Price,test_size=0.2,random_state=21)

#fitting the model with training set
linreg.fit(X_train,y_train)

#predicting target (Price) with test inputs
y_pred=linreg.predict(X_test)

#printing training accuracy score
print("Training Score is: ")
print(linreg.score(X_train,y_train))

#printing test accuracy score
print("\nTest Score is: ")
print(linreg.score(X_test,y_test))

# plotting test label data and predicted data
plt.subplot(2,1,1)
plt.plot(y_test.values)
plt.subplot(2,1,2)
plt.plot(y_pred)

#updating random value of test data
y_test.update(pd.Series([100],index=[321]))

#printing test accuracy score
print("\nTest Score after changing 1 of Test Input values is: ")
print(linreg.score(X_test,y_test))