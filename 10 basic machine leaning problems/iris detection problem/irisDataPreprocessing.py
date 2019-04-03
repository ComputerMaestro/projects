##Data preprocessing
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt

data = pd.read_csv('irisdataset.csv')
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
##Shuffling rows randomly to evenly distribute the classes on the tempframe (3 times)
##By np.random.permutation the temp will be converted to numpy array
temp = data
for i in range(3):
      temp = np.random.permutation(temp)
temp = np.concatenate((temp, np.zeros((149, 3))), axis=1)
temp[temp[:, 4] == 'Iris-setosa', 5], temp[temp[:, 4] == 'Iris-versicolor', 6], temp[temp[:, 4] == 'Iris-virginica', 7] = 1, 1, 1
temp = np.float64(temp[:, [0,1,2,3,5,6,7]])
X, Y = np.concatenate((np.ones((100, 1)), temp[:100, :4]), axis=1), temp[:100, 4:]
Xcv, Ycv = np.concatenate((np.ones((25, 1)), temp[100:125, :4]), axis=1), temp[100:125, 4:]
Xtest, Ytest = np.concatenate((np.ones((temp[125:, :4].shape[0], 1)), temp[125:, :4]), axis=1), temp[125:, 4:]
del temp
