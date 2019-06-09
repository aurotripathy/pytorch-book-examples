"""
Single layer model
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

dataset = [((-2, -1), 0), ((-2, 1), 1), ((-1, -1.5), 0),
           ((1, 1), 1), ((1.5, -0.5), 1), ((2, -2), 0)]

slp_model.train()
for epoch in range(1000):
    for i, data in enumerate(dataset):
        X = torch.tensor([data[0]],
                         dtype=torch.float, requires_grad=True)
        y = torch.tensor([data[1]],
                         dtype=torch.float, requires_grad=True)
                                       
        optimizer.zero_grad()
        y_pred = slp_model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print('Epoch {}, loss:{}'.format(epoch, loss))

print('Model params:', list(slp_model.parameters()))  # https://graphsketch.com/

# Test
slp_model.eval()
out = slp_model(torch.FloatTensor([dataset[0][0]]))
print('in {}, out prob {}, class {}'.format(dataset[0][0], out, '0' if out < 0.5 else '1'))
out = slp_model(torch.FloatTensor([dataset[1][0]]))
print('in {}, out prob {}, class {}'.format(dataset[1][0], out, '0' if out < 0.5 else '1'))
out = slp_model(torch.FloatTensor([dataset[2][0]]))
print('in {}, out prob {}, class {}'.format(dataset[2][0], out, '0' if out < 0.5 else '1'))
out = slp_model(torch.FloatTensor([dataset[3][0]]))
print('in {}, out prob {}, class {}'.format(dataset[3][0], out, '0' if out < 0.5 else '1'))
