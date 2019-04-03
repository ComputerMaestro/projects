import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Ytrain = pd.read_csv('trainset.csv')

##Getting the Series of strings of image pixel number by pop() method
##Splitting the strings to lists of the individual number strings
##converting the Series of lists to list of lists by tolist() method
##Then converting this list of lists to dataframe
##Atlast converting this dataframe dtypes from string to integer of 8 bits
Xtrain = pd.DataFrame(Ytrain.pop('Image').str.split(' ').tolist()).astype(np.uint8).values
Xtrain,Ytrain = np.uint8(np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)), Ytrain.values
m, n, o = Xtrain.shape[0], Xtrain.shape[1]-1, Ytrain.shape[1]
epsilon = np.sqrt(6/(9216+30))

##Random initialisation
weights1 = np.random.rand(100, n+1)*epsilon
weights2 = np.random.rand(o, 101)*epsilon
##weights1 = pd.read_csv('weights1.csv', header=None).values
##weights2 = pd.read_csv('weights2.csv', header=None).values

alpha = 0.00000000001
J = np.zeros((m, 1))
##Training
for i in range(m):
      #print(i)
      x = Xtrain[i, :].reshape((9217, 1))
      a = np.concatenate((np.ones((1, 1)), np.dot(weights1, x)))
      h = np.dot(weights2, a)
      J[i] = 0.5*(np.sum(np.square(h - Ytrain[i, :].reshape((o, 1)))))
      dO = h-Ytrain[i, :].reshape((o, 1))
      delta2 = np.repeat(a, [o], axis=1).T*dO
##      print(delta2)
      da = np.dot(weights2[:, 1:].T, dO)
      delta1 = np.repeat(da, [n+1], axis=1)*(x.T)
      weights2 = weights2 - alpha*delta2
      weights1 = weights1 - alpha*delta1

w1 = pd.DataFrame(weights1)
w2 = pd.DataFrame(weights2)

w1.to_csv('weights1.csv', header=False, index=False)
w2.to_csv('weights2.csv', header=False, index=False)

plt.plot(range(m), J)
plt.show()
