"""
Single layer model
"""
import torch
import torch.nn as nn
from torch.nn import Linear, Sigmoid
import torch.optim as optimizer
from torch.autograd import Variable
from pudb import set_trace
import torch.nn.functional as F


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
                                       
        set_trace()
        optimizer.zero_grad()
        y_pred = slp_model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print('Epoch {}, loss:{}'.format(epoch, loss))

# set_trace()
print('Model params:', list(slp_model.parameters()))
# Test
slp_model.eval()
out = slp_model(torch.FloatTensor([(-2, -1)]))
print(out)
out = slp_model(torch.FloatTensor([(-2, 1)]))
print(out)
out = slp_model(torch.FloatTensor([(-1, -1.5)]))
print(out)
out = slp_model(torch.FloatTensor([(1, 1)]))
print(out)

