# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Impporting Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training decision tree model on whole data set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(x,y) 

#Predicting a new result
print(regressor.predict([[6.5]]))

#Visualize Polynomial model results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid) , 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid , regressor.predict(x_grid) , color = 'blue')
plt.title('Truth or bluffDecision Tree Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()