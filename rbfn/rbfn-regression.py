"""
RBF with unsupervised hidden layer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import kmeans, initialize_weights
torch.manual_seed(777)

class RbfNet(nn.Module):
    """
    Radial Basis Function network
    """
    def __init__(self, centers, sigma):
        super(RbfNet, self).__init__()
        self.num_centers = len(centers)
        self.centers = centers
        self.sigma = sigma
        self.linear = nn.Linear(self.num_centers, 1, bias=True)
        initialize_weights(self)

    def kernel_func(self, X):
        """
        Gaussian radial basis function
        x is the input
        centers is the cluster centers
        sigma is the std deviation
        """
        return torch.exp(-1 / (2 * self.sigma ** 2) * (X - self.centers)**2)


    def forward(self, X):
        radial_val = self.kernel_func(X)
        score = self.linear(radial_val)
        return score[0]


class RbfImplement():
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, epochs=100, inferStds=True):
        self.k = k
        self.epochs = epochs
        self.inferStds = inferStds
        self.lr = lr

        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
            print('centers, stds', self.centers, self.stds)
        else:
            # use a fixed std 
            self.centers, _ = kmeans(X, self.k)
            # new_kmeans = KMeans(self.k, random_state=0).fit(X.reshape(-1,1))
            # self.centers = new_kmeans.cluster_centers_.flatten()
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        self.centers = torch.from_numpy(self.centers).float()
        self.stds = torch.from_numpy(self.stds).float()
        self.model = RbfNet(self.centers, self.stds)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fun = nn.MSELoss()


    def train(self, X, y):
        self.model.train()
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                self.optimizer.zero_grad()          # Zero Gradient Container
                y_pred = self.model(X[i])           # Forward Propagation
                loss = self.loss_fun(y_pred, y[i])  # compute loss
                loss.backward()                     # compute gradient
                self.optimizer.step()               # gradient update
        print('Model Params', list(self.model.parameters()))

    def test(self, X):
        self.model.eval()
        X = torch.tensor(X).float()
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.model(X[i]).detach().numpy())  # Forward Propagation
        return y_pred


# input samples with noise added
NUM_SAMPLES = 100
X = np.random.uniform(0., 1., NUM_SAMPLES)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = np.sin(2 * np.pi * X)  + noise

rbfn = RbfImplement(lr=1e-2, k=10, inferStds=False)
rbfn.train(X, y)
y_pred = rbfn.test(X)

plt.plot(X, y, '^', label='ground-truth')
plt.plot(X, y_pred, 'o', label='RBF-Net predicted')
plt.legend()
plt.tight_layout()
plt.show()
