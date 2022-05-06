# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Impporting Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Reshaping y in 2d array
y = y.reshape((len(y) , 1))

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#Traning the SVR model on whole data set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

#Predicting a new result
res = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
print(res)

#Visualizing the SVR results
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x) , sc_y.inverse_transform(regressor.predict(x)) , color = 'blue')
plt.title('Truth or bluff(SVR Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#visualize in high resolution
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid) , 1))
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid , sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))) , color = 'blue')
plt.title('Truth or bluff(SVR Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

