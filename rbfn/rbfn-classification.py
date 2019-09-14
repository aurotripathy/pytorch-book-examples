"""
Classification with unsupervised RBF hidden layer
"""
import math
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from utils import display, display_misclassified, get_dataset_labels, display_loss_accuracy

torch.manual_seed(777)

class RbfNet(nn.Module):
    """
    Radial Basis Function network
    """
    def __init__(self, centers, sigma, nb_classes=2):
        super(RbfNet, self).__init__()
        self.num_centers = len(centers)
        self.centers = centers
        self.sigma = sigma
        self.linear = nn.Linear(self.num_centers, nb_classes, bias=True)

    def kernel_func(self, X):
        """
        Gaussian radial basis function
        - X is the input
        - centers is the cluster centers
        - sigma is the std deviation
        """
        h = np.zeros(self.num_centers)
        for i in range(self.num_centers):
            h[i] = math.exp((-1 / (2 * self.sigma ** 2)) * np.linalg.norm(X - self.centers[i])**2)
        return h

    def forward(self, X):
        radial_val = torch.tensor(self.kernel_func(X), requires_grad=True).float()
        score = self.linear(radial_val)
        score.unsqueeze_(0)
        return score


class RbfImplement():
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, num_rbf_neurons=100, lr=0.01, epochs=100):
        self.num_centers = num_rbf_neurons
        self.epochs = epochs
        self.lr = lr

    def train(self, Xs, ys):
        """
        Training an RBFN consists of selecting three sets of parameters:
        The first two are unsupervised and the last is supervised.
        1. The prototypes (centers). Unsupervised
        2. Beta coefficient for each of the RBF neurons. Unsupervised
        3. The matrix of output weights between the RBF neurons and the output nodes. Supervised.
        """
        kmeans = KMeans(n_clusters=self.num_centers)
        kmeans.fit(dataset)
        centers = torch.from_numpy(kmeans.cluster_centers_).float()

        # calc d_max, the maximum distance between two hidden neurons
        f_centers = kmeans.cluster_centers_.flatten()
        d_max = max([np.abs(c1 - c2) for c1 in f_centers for c2 in f_centers])
        sigma = d_max / np.sqrt(2*self.num_centers)  # the sigma heuristic

        model = RbfNet(centers, sigma)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_func = nn.CrossEntropyLoss()  # combines nn.LogSoftmax() and nn.NLLLoss()

        model.train()
        loss_accuracy = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            for X, y in zip(Xs, ys):
                X = torch.tensor([X]).float()
                y = torch.tensor([y]).long()
                optimizer.zero_grad()       # Zero Gradient Container
                y_pred = model(X)           # Forward Propagation
                loss = loss_func(y_pred, y) # compute loss
                epoch_loss += loss
                loss.backward()             # compute gradient
                optimizer.step()            # gradient update
            avg_loss = epoch_loss / len(Xs)
            accuracy = sklearn.metrics.accuracy_score(ys, rbfn.test(dataset, model))
            print('Avg loss: {:.4f}, Accuracy: {:.4f}:'.format(avg_loss, accuracy))
            loss_accuracy.append((avg_loss, accuracy))
        return model, loss_accuracy

    def test(self, Xs, model):
        model.eval()
        y_pred = []
        for X in Xs:
            X = torch.tensor([X]).float()
            predicted = torch.argmax(model(X))
            y_pred.append(predicted)
        return y_pred


dataset, labels = get_dataset_labels()
display(dataset, labels, 'Scatter Plot of Ground Truth')
rbfn = RbfImplement(lr=1e-2, num_rbf_neurons=100, epochs=20)
trained_model, loss_accuracy = rbfn.train(dataset, labels)
predicted_labels = rbfn.test(dataset, trained_model)
print('Accuracy:', sklearn.metrics.accuracy_score(labels, predicted_labels))
display(dataset, predicted_labels, 'Scatter Plot of Predicted')
display_misclassified(dataset, labels, predicted_labels, 'Scatter Plot of Misclassified')
display_loss_accuracy(loss_accuracy, 'Loss Accuracy Plot')

# References: http://www.charuaggarwal.net/Chap5slides.pdf
