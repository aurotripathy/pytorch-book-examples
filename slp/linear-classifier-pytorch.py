"""
Binary Classifier with Linear layer and Back Propagation of Errors
"""
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.optim as optimizer
from utils import display_loss, display_points, plot_points_line_slope_intercept

class LinearClassifier(nn.Module):
    """ One Linear layer """
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fully_connected = Linear(2, 1)


    def forward(self, x):
        return self.fully_connected(x)  # WX + b


model = LinearClassifier()
criterion = nn.MSELoss()
optimizer = optimizer.SGD(model.parameters(), lr=0.01, momentum=0.5)

train_set = [((-2, -1), 0), ((-2, 1), 1), ((-1, -1.5), 0),
             ((1, 1), 1), ((1.5, -0.5), 1), ((2, -2), 0)]
display_points([sample[0] for sample in train_set],
               [sample[1] for sample in train_set], "Data")

model.train()
loss_over_epochs = []
for epoch in range(50):
    epoch_loss = 0
    for train_data in train_set:
        X = torch.tensor([train_data[0]], dtype=torch.float, requires_grad=True)
        y = torch.tensor([train_data[1]], dtype=torch.float, requires_grad=True)

        optimizer.zero_grad() # Zero out for each batch
        y_pred = model(X)     # Forward Propagation
        loss = criterion(torch.squeeze(y_pred, 1), y)  # Compute loss
        loss.backward()       # Compute gradient
        optimizer.step()      # Update model paramters
        epoch_loss += loss

    loss_over_epochs.append(epoch_loss)
    print('Epoch {}, Epoch loss:{}'.format(epoch, epoch_loss))

display_loss(loss_over_epochs, "Loss Plot")
print('Model params:', list(model.parameters()))  # https://graphsketch.com/

# Test on unseen data
test_set = [(0.5, 0.5), (1.5, 1.5), (2, -1), (-3, -3)]
model.eval()
for test_data in test_set:
    out = model(torch.tensor([test_data], dtype=torch.float, requires_grad=False))
    print('Data in: {}, Out Prob: {}, Predicted Class {}'.format(test_data, out, '0' if out < 0.5 else '1'))

(w1, w2) = model.fully_connected.weight.data.numpy()[0]
b = model.fully_connected.bias.data.numpy()[0]

plot_points_line_slope_intercept([sample[0] for sample in train_set],
                                 [sample[1] for sample in train_set],
                                 -w1/w2, -b, 'Dividing Line Back Propagation')


    
# Test on training data
for train_data in train_set:
    prob = model(torch.tensor([train_data[0]], dtype=torch.float, requires_grad=False))
    label = 0 if prob < 0.5 else 1
    verdict = 'correct' if label == train_data[1] else 'wrong'
    print('Data in: {}, Actual Class: {} Out Score: {}, Predicted Class {}: {}'.format(train_data, train_data[1], prob, label, verdict))
