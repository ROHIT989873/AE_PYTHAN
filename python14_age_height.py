# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:00:56 2021

@author: RM
"""

import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/RM/Downloads/AGE_HEIGHT.csv')
print(type(dataset))

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Note: The parameter 'random_state' is used to randomly bifurcate the dataset into training &
#testing datasets. That number should be supplied as arguments to parameter 'random_state'
#which helps us get the max accuracy. And that number is decided by hit & trial method.

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Calculating the coefficients:
    
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Accuracy of the model

#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#Create a DataFrame
df1 = {'Actual Applicants':y_test,
'Predicted Applicants':y_pred}
df1 = pd.DataFrame(df1,columns=['Actual Applicants','Predicted Applicants'])
print(df1)

# Visualising the predicted results
line_chart1 = plt.plot(X_test,y_pred, '--', c ='red')
line_chart2 = plt.plot(X_test,y_test, ':', c='blue')


# Scatter plot in Python:

Age=[18,19,20,21,22,23,24,25,26,27,28,29]
Height=[76.1,77,78.1,78.2,78.8,79.7,79.9,81.1,81.2,81.8,82.8,83.5]
plt.scatter(Age,Height , c='r',marker='*')
plt.xlabel('Age', fontsize=16)
plt.ylabel('Height', fontsize=16)
plt.title('scatter plot -Age vs Height',fontsize=20)
