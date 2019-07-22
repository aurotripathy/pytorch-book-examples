"""
ADDED Manual
Binary Classifier (Linear layer followed by logistic avtivation)
"""
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.optim as optimizer
from utils import display_loss, display_points
from pudb import set_trace
import numpy as np

torch.manual_seed(0)

class LinearClassifier(nn.Module):
    """ One Linear layer """
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fully_connected = Linear(2, 1)


    def forward(self, x):
        return self.fully_connected(x)  # WX + b

lr = 0.01  # learning rate

model = LinearClassifier()
print('Model params', list(model.parameters()))
criterion = nn.MSELoss()
optimizer = optimizer.SGD(model.parameters(), lr=lr, momentum=0.5)

train_set = [((-2, -1), 0), ((-2, 1), 1), ((-1, -1.5), 0),
             ((1, 1), 1), ((1.5, -0.5), 1), ((2, -2), 0)]
display_points([sample[0] for sample in train_set],
               [sample[1] for sample in train_set], "Data")

set_trace()
manual_W = model.fully_connected.weight.data.numpy()[0].copy()  # [np.random.randn(), np.random.randn()]
manual_b = model.fully_connected.bias.data.numpy()[0].copy()  # np.random.randn()

model.train()

loss_over_epochs = []
manual_loss_over_epochs = []

for epoch in range(50):
    epoch_loss = 0
    manual_epoch_loss = 0
    for train_data in train_set:
        X = torch.tensor([train_data[0]], dtype=torch.float, requires_grad=True)
        y = torch.tensor([train_data[1]], dtype=torch.float, requires_grad=True)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(torch.squeeze(y_pred, 1), y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss

        # do the training manually
        manual_y_pred = np.dot(train_data[0], manual_W) + manual_b
        manual_error = (train_data[1] - manual_y_pred)
        manual_loss = manual_error ** 2
        manual_epoch_loss += manual_loss

        # Computes gradients for both W and b parameters
        W_grad = -2 * (np.array(train_data[0]) * manual_error).mean()
        b_grad = -2 * manual_error.mean()
       
        # Updates parameters using gradients and the learning rate
        manual_W -= lr * W_grad
        manual_b -= lr * b_grad
        
    loss_over_epochs.append(epoch_loss)
    manual_loss_over_epochs.append(manual_epoch_loss)
    print('Epoch {}, Pytorch Epoch loss:{}, Manual Epoch loss:{}'.format(epoch, epoch_loss, manual_epoch_loss))


display_loss(manual_loss_over_epochs, "Loss Plot Manual Calculations")
print('Model params:', list(model.parameters()))

# Test on unseen data
test_set = [(0.5, 0.5), (1.5, 1.5), (2, -1), (-3, -3)]
model.eval()
for test_data in test_set:
    out = model(torch.tensor([test_data], dtype=torch.float, requires_grad=False))
    print('Data in: {}, Out Prob: {}, Predicted Class {}'.format(test_data, out, '0' if out < 0.5 else '1'))

# Test on training data
for train_data in train_set:
    prob = model(torch.tensor([train_data[0]], dtype=torch.float, requires_grad=False))
    label = 0 if prob < 0.5 else 1
    verdict = 'correct' if label == train_data[1] else 'wrong'
    print('Data in: {}, Actual Class: {} Out Prob: {}, Predicted Class {}: {}'.format(train_data, train_data[1], prob, label, verdict))
