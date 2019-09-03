import numpy as np

class MADALINE:
    def __init__(self, in_layer_sz, hidden_layer_sz, threshold=0):
        self.W1 = 10*np.random.randn(hidden_layer_sz, in_layer_sz)
        self.sum = np.zeros((hidden_layer_sz, 1))
        self.W2 = 10*np.random.randn(1, hidden_layer_sz)
        self.b1 = 10*np.random.randn(hidden_layer_sz, 1)
        self.b2 = 10*np.random.randn(1, 1)
        self.t = threshold
        self.vactivation = np.vectorize(self.activation)

    def forward(self, x):
        """Computes forward pass for the Madaline"""
        self.sum = np.matmul(self.W1, x) + self.b1
        h = self.vactivation(self.sum)
        z = np.dot(self.W2, h) + self.b2
        o = self.vactivation(z)
        return o

    def activation(self, s):
        """ACtivation funcion for Madaline f(sum) >= threshold return 1
        else if f(sum) < threshold return -1"""
        return 1 if (s >= self.t) else -1

    def train(self, data, learning_rate, epochs):
        """Training function for Madaline"""
        for epoch in range(epochs):
            accuracy = 0
            for (x, y) in data:
                y = y[0, 0]
                h = self.forward(x)
                if h[0, 0] != y:
                    if y == 1:
                        square = self.sum**2
                        q = square == np.amin(square)
                    else:
                        q = self.sum > 0
                    self.b1[q] += learning_rate*(y-self.sum[q])
                    self.W1[np.repeat(q, self.W1.shape[1], axis=1)] += (learning_rate*(y-self.sum[q].reshape((-1, 1)))*(x.T)).reshape(-1)
                else:
                    accuracy += 1
                    print("h:", h[0, 0], " y:", y)
            print("Epoch: ", epoch+1)
            print("Accuracy:", (accuracy/len(data))*100)

    def predict(self, x):
        """Used to predict y after training to test the model"""
        sum = np.matmul(self.W1, x) + self.b1
        h = self.vactivation(sum)
        z = np.matmul(self.W2, h) + self.b2
        o = self.vactivation(z)
        return o[0, 0]