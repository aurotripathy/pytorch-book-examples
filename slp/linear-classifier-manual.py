"""
A binary classifier with a lingle linear layer
"""
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.optim as optimizer
from utils import display_loss, display_points, plot_points_line_slope_intercept
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

manual_W = model.fully_connected.weight.data.numpy()[0].copy()  # init manual model the same as PyTorch
manual_b = model.fully_connected.bias.data.numpy()[0].copy()  # init manual model the same as PyTorch

model.train()

loss_over_epochs = []
manual_loss_over_epochs = []

for epoch in range(100):
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

        # Computes gradients for W and b parameters
        manual_W_grad = -2 * (np.array(train_data[0]) * manual_error).mean()
        manual_b_grad = -2 * manual_error.mean()
       
        # Updates parameters using gradients and the learning rate
        manual_W -= lr * manual_W_grad
        manual_b -= lr * manual_b_grad
        
    loss_over_epochs.append(epoch_loss)
    manual_loss_over_epochs.append(manual_epoch_loss)
    print('Epoch {}, Pytorch loss:{}, Manual loss:{}'.format(epoch, epoch_loss, manual_epoch_loss))


display_loss(manual_loss_over_epochs, "Loss Plot Manual Calculations")
print('Model params:', list(model.parameters()))

(w1, w2) = model.fully_connected.weight.data.numpy()[0]
b = model.fully_connected.bias.data.numpy()[0]

plot_points_line_slope_intercept([sample[0] for sample in train_set],
                                 [sample[1] for sample in train_set],
                                 -w1/w2, -b, 'Dividing Line')

set_trace()
print('Model params (computed manually):', 'W', manual_W, 'b', manual_b)
plot_points_line_slope_intercept([sample[0] for sample in train_set],
                                 [sample[1] for sample in train_set],
                                 -manual_W[0]/manual_W[1], -manual_b,
                                 'Dividing Line, manual computation')

# Test on unseen data
test_set = [((0.5, 0.5), 1), ((-1, -2), 0), ((1.5, 1.5), 1), ((2.5, -0.5), 1), ((-2.75, -2.75), 0), ((-2.0, -0.5), 0)]
model.eval()
for test_data in test_set:
    out = model(torch.tensor([test_data[0]], dtype=torch.float, requires_grad=False))
    predicted_class = '0' if out < 0.5 else '1'
    print('Input: {}, Out Prob: {}, Predicted Class {}'.format(test_data, out, predicted_class))

for test_data in test_set:
    manual_y_pred = np.dot(test_data[0], manual_W) + manual_b
    manual_predicted_class = '0' if manual_y_pred < 0.5 else '1'
    print('Manual: Input: {}, Out Score: {}, Predicted Class {}'.format(test_data,
                                                                        manual_y_pred,
                                                                        manual_predicted_class))
plot_points_line_slope_intercept([sample[0] for sample in test_set],
                                 [sample[1] for sample in test_set],
                                 -manual_W[0]/manual_W[1], -manual_b,
                                 'Dividing Line, manual computation')

# Test on training data
for train_data in train_set:
    score = model(torch.tensor([train_data[0]], dtype=torch.float, requires_grad=False))
    label = 0 if score < 0.5 else 1
    verdict = 'correct' if label == train_data[1] else 'wrong'
    print('Input: {}, Actual: {} Out Score: {}, Predicted {}: {}'.format(train_data[0],
                                                                         train_data[1],
                                                                         score,
                                                                         label, verdict))
