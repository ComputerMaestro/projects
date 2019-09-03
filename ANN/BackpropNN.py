import numpy as np

class BackpropNN:
    def __init__(self, in_sz, hid_sz, out_sz, slope):
            self.Wi = 10*np.random.randn(hid_sz, in_sz)
            self.bi = 10*np.random.randn(hid_sz, 1)
            self.Wh = 10*np.random.randn(out_sz, hid_sz)
            self.bh = 10*np.random.randn(out_sz, 1)
            self.s = slope

    def forward(self, x):
        self.zi = np.dot(self.Wi, x) + self.bi
        self.a = self.sigmoid(self.zi)
        self.zh = np.dot(self.Wh, self.a) + self.bh
        return self.sigmoid(self.zh)

    def sig_derivative(self, x):
        return self.s*np.exp((-self.s)*x)/((1+np.exp((-self.s)*x))**2)

    def sigmoid(self, x):
        return 1/(1+np.exp((-self.s)*x))

    def loss(self, x, y):
        return (np.sum(x - y)**2)


    def train(self, data, learning_rate, epochs=1):
        for epoch in range(epochs):
            dWh, dbh, dWi, dbi, m, l = np.zeros(self.Wh.shape), np.zeros(self.bh.shape), np.zeros(self.Wi.shape), np.zeros(self.bi.shape), len(data), 0
            for (x, y) in data:
                y_hat = self.forward(x)
                l += self.loss(y_hat, y)
                dzh = 2*(y_hat - y)*self.sig_derivative(self.zh)
                dWh += dzh*self.a.T
                dbh += dzh
                da = np.sum(dzh*self.Wh, axis=0).reshape((-1, 1))
                dzi = da*self.sig_derivative(self.zi)
                dWi += dzi*x.T
                dbi += dzi
            l /= m
            dWh /= m
            dbh /= m
            dWi /= m
            dbi /= m
            self.Wi -= learning_rate*dWi
            print(dWi)
            self.bi -= learning_rate*dbi
            self.Wh -= learning_rate*dWh
            self.bh -= learning_rate*dbh
            print("Epoch:", epoch+1)
            print("Loss:", l)
            print("\n#----------------------------------------------------------------#\n")

    def predict(self, x):
        z = np.dot(self.Wi, x) + self.bi
        a = self.sigmoid(z)
        z = np.dot(self.Wh, a) + self.bh
        y = self.sigmoid(z)
        return y