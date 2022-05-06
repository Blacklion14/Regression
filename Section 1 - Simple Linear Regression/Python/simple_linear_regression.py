# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Impporting Dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Spliting Data Into Training And Test Set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size = 0.2 , random_state = 0)

#Training the model on training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)

#Predicting the test results
y_pred = regressor.predict(x_test)

#Visualising the training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#To print the salary of a person with 12year exp.
print(regressor.predict([[12]]))

#Getting the final linear regression equation (y=b0 + b1x) with the values of the coefficients 
print(regressor.coef_)
print(regressor.intercept_)


























