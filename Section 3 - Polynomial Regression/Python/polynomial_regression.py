# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Impporting Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the model on data for linear regressor
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x , y)

#Training the model on data for polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
x_poly = poly_reg.fit_transform(x)

poly_lin_reg = LinearRegression()
poly_lin_reg.fit(x_poly , y)

#Visualize linear model results
plt.scatter(x, y, color = 'red')
plt.plot(x , lin_reg.predict(x) , color = 'blue')
plt.title('Truth or bluff(linear Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualize Polynomial model results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid) , 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid , poly_lin_reg.predict(poly_reg.fit_transform(x_grid)) , color = 'blue')
plt.title('Truth or bluff(Polynomial Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#predicting linear model
print(lin_reg.predict([[6.5]]))

#predicting linear model
print(poly_lin_reg.predict(poly_reg.fit_transform([[6.5]])))



























 