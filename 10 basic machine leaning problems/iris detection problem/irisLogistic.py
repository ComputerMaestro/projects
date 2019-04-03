import numpy as np, pandas as pd, os, matplotlib.pyplot as plt

##Data Preprocessing
import irisDataPreprocessing as idp

##Multivariate logistic regression
def sigmoid(z):
      return 1/(1 + np.exp(-z))

i, m, alpha, Theta = 0, idp.X.shape[0], 0.1, np.zeros((5, 3))
while i < 5000:
      h = sigmoid(np.dot(idp.X, Theta))
      ##First Classifier For Setosa
      J = (-1/m)*(np.sum((idp.Y[0]*np.log(h[0])+(1-idp.Y[0])*np.log(1-h[0])))+
                  np.sum((idp.Y[1]*np.log(h[1])+(1-idp.Y[1])*np.log(1-h[1])))+
                  np.sum((idp.Y[2]*np.log(h[2])+(1-idp.Y[2])*np.log(1-h[2]))))
      print(J)
      Theta = Theta - (alpha/m)*(np.dot(idp.X.T, h - idp.Y))
      i += 1
