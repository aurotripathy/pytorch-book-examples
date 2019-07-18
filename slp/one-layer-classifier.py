"""
Binary Classifier (Linear layer followed by logistic avtivation)
"""
import torch
import torch.nn as nn
from torch.nn import Linear, Sigmoid
import torch.optim as optimizer


class LinearClassifier(nn.Module):
    """ Linear layer followed by Sigmoid non-linearity"""
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fully_connected = Linear(2, 1)
        self.sigmoid = Sigmoid()


    def forward(self, x):
        return self.sigmoid(self.fully_connected(x))


model = LinearClassifier()
criterion = nn.MSELoss()
optimizer = optimizer.SGD(model.parameters(), lr=0.01, momentum=0.5)

train_set = [((-2, -1), 0), ((-2, 1), 1), ((-1, -1.5), 0),
             ((1, 1), 1), ((1.5, -0.5), 1), ((2, -2), 0)]

model.train()
for epoch in range(1000):
    epoch_loss = 0
    for train_data in train_set:
        X = torch.tensor([train_data[0]], dtype=torch.float, requires_grad=True)
        y = torch.tensor([train_data[1]], dtype=torch.float, requires_grad=True)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(torch.squeeze(y_pred, 1), y)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print('Epoch {}, Epoch loss:{}'.format(epoch, epoch_loss))

print('Model params:', list(model.parameters()))  # https://graphsketch.com/

# Test on unseen data
test_set = [(0.5, 0.5), (1.5, 1.5), (2, -1), (-3, -3)]
model.eval()
for test_data in test_set:
    out = model(torch.tensor([test_data], dtype=torch.float, requires_grad=False))
    print('Data in: {}, Out prob: {}, class {}'.format(test_data, out, '0' if out < 0.5 else '1'))

# Test on training data
for train_data in train_set:
    prob = model(torch.tensor([train_data[0]], dtype=torch.float, requires_grad=False))
    label = 0 if prob < 0.5 else 1
    verdict = 'correct' if label == train_data[1] else 'wrong'
    print('Data in: {}, Out prob: {}, class {}: {}'.format(train_data, prob, label, verdict))
