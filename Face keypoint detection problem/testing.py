import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

Ytest = pd.read_csv('test.csv')
Xtest = pd.DataFrame(Ytest.pop('Image').str.split(' ').tolist()).astype(np.uint8).values
Xtest,Ytest = np.uint8(np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)), Ytest.values
mtest, o = Xtest.shape[0], 30

weights1 = pd.read_csv('avgweights1.csv', header=None).values
weights2 = pd.read_csv('avgweights2.csv', header=None).values

E = np.zeros((mtest, 1))
for i in range(mtest):
      print(i)
      x = Xtest[i, :].reshape((9217, 1))
      y = Ytest[i, :].reshape((o, 1))
      nans = np.where(pd.isnull(y))[0]
      w2 = np.delete(weights2, nans, axis=0)
      a = np.concatenate((np.ones((1, 1)), np.dot(weights1, x)))
      predict = np.dot(w2, a)
      E[i] = np.sum(np.square(predict - np.delete(y, nans, axis=0)))

plt.plot(range(340), E, color='r')
plt.show()
      
x = Xtest[6, :]
y = np.int32(Ytest[1, :]).reshape((30, 1))
nans = np.where(pd.isnull(y))[0]
w2 = np.delete(weights2, nans, axis=0)
a = np.concatenate((np.ones((1, 1)), np.dot(weights1, x.reshape((9217, 1)))))
predict = np.int32(np.dot(w2, a))
img = cv2.cvtColor(np.uint8(x[1:].reshape((96, 96))).copy(), cv2.COLOR_GRAY2BGR)
for i in range(15):
      img = cv2.circle(img, (predict[i*2], predict[i*2+1]), 2, (255, 0, 0), -1)
      img = cv2.circle(img, (y[i*2], y[i*2+1]), 2, (0, 0, 255), -1)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
