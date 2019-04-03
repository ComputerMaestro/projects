##Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

##Data pre-processing
Ytrain = pd.read_csv('train.csv')
Xtrain = pd.DataFrame(Ytrain.pop('Image').str.split(' ').tolist()).astype(np.uint8).values
Xtrain,Ytrain = np.uint8(np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)), Ytrain.values
m, n, o = Xtrain.shape[0], Xtrain.shape[1]-1, Ytrain.shape[1]

alpha = 0.00000000001
epsilon = np.sqrt(6/(9216+100))

##weights2 rearranging function
def weights(nans, weights2, w2):
      j = 0
      for i in range(weights2.shape[0]):
            if i in nans: continue
            weights2[i, :] = w2[j, :]
            j += 1
      return weights2

J = np.zeros((m, 1, 10))
##Training NN
weights1list,weights2list = [], []
for epoch in range(10):
      
      ##Random Initialization
      weights1 = np.random.rand(100, n+1)*epsilon
      weights2 = np.random.rand(o, 101)*epsilon
      
      for i in range(m):
            print(epoch, i)
            x = Xtrain[i, :].reshape((9217, 1))
            y = Ytrain[i, :].reshape((o, 1))
            nans = np.where(pd.isnull(y))[0]
            w2 = np.delete(weights2, nans, axis=0)
            rem = o-nans.shape[0]

            ##Forward propagation
            a = np.concatenate((np.ones((1, 1)), np.dot(weights1, x)))
            h = np.dot(weights2, a)
            J[i, 0, epoch] = 0.5*(np.sum(np.square(h - y)))

            ##Backpropagation
            dO = np.delete(h-Ytrain[i, :].reshape((o, 1)), nans, axis=0)
            delta2 = np.repeat(a, [rem], axis=1).T*dO
            da = np.dot(w2[:, 1:].T, dO)
            delta1 = np.repeat(da, [n+1], axis=1)*(x.T)

            ##Changing weights
            w2 = w2 - alpha*delta2
            weights2 = weights(nans, weights2, w2)
            weights1 = weights1 - alpha*delta1
      weights1list.append(weights1)
      weights2list.append(weights2)
