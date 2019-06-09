"""
Single Layer Binary Classifier
"""
import torch
import torch.nn as nn
from torch.nn import Linear, Sigmoid
import torch.optim as optimizer


class SLP(nn.Module):
    def __init__(self):
        super(SLP, self).__init__()
        self.fully_connected = Linear(2, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fully_connected(x))


slp_model = SLP()
criterion = nn.MSELoss()
optimizer = optimizer.SGD(slp_model.parameters(), lr=0.01, momentum=0.5)

train_set = [((-2, -1), 0), ((-2, 1), 1), ((-1, -1.5), 0),
             ((1, 1), 1), ((1.5, -0.5), 1), ((2, -2), 0)]

slp_model.train()
for epoch in range(1000):
    for train_data in enumerate(train_set):
        X = torch.tensor([train_data[0]], dtype=torch.float, requires_grad=True)
        y = torch.tensor([train_data[1]], dtype=torch.float, requires_grad=True)
                                       
        optimizer.zero_grad()
        y_pred = slp_model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print('Epoch {}, loss:{}'.format(epoch, loss))

print('Model params:', list(slp_model.parameters()))  # https://graphsketch.com/

# Test on unseen data
test_set = [(0.5, 0.5), (1.5, 1.5), (2, -1), (-3, -3)]
slp_model.eval()
for data in test_set:
    out = slp_model(torch.tensor([data], dtype=torch.float, requires_grad=False))
    print('Data in: {}, out prob {}, class {}'.format(data, out, '0' if out < 0.5 else '1'))

# Test on train data
for train_data in train_set:
    prob = slp_model(torch.tensor([train_data[0]], dtype=torch.float, requires_grad=False))
    label = 0 if prob < 0.5 else 1
    verdict = 'correct' if label == train_data[1] else 'wrong'
    print('Data in: {}, out prob {}, class {}: {}'.format(data, prob, label, verdict))
