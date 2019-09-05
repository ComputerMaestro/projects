"""
KOHONEN
"""
import numpy as np

class KohonenNN:
    def __init__(self, in_units, K):
        self.W = np.random.randn(in_units, K)

    def forward(self, x):
        """Calculates Euclidean distance between the input and all the weights"""
        E = np.sum((self.W-x)**2, axis=0).reshape(-1, 1)
        return E

    def cluster(self, data, learning_rate, R, R_decay=0.5, lr_decay= 0.5, epochs=1):
        """This function applies Kohonen learning criteria to cluster the given inputs into groups"""
        clusters = [[] for i in range(self.W.shape[1])]
        for epoch in range(epochs):
            for x in data:
                x = x.reshape((-1, 1))
                E = self.forward(x)
                indices = E.argsort(axis=0)[:R, 0]
                self.W[:, indices] += learning_rate*(x-self.W[:, indices])
                if epoch == (epochs-1):
                    clusters[indices[0]].append(x)
            R = int(np.ceil(R*R_decay))
            learning_rate *= lr_decay
            print("Epoch:", epoch+1)
            print("Overall Euclidean distance value:", np.sum(E))
            print("Topological param:", R)
            print("Learning rate:", learning_rate)
            print("\n#----------------------------------------------------------------#\n")
        print("Clusters:")
        for k in range(len(clusters)):
            print("cluster ", k+1)
            print(clusters[k])