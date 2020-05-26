import torch
import torch.nn as nn
from torch.autograd import Variable

class MyActivationFunction(nn.Module):

    def __init__(self, mean=0, std=1, min=0.1, max=0.9):
        super(MyActivationFunction, self).__init__()
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max

    def forward(self, x):
        gauss = torch.exp((-(x - self.mean) ** 2)/(2* self.std ** 2))
        return torch.clamp(gauss, min=self.min, max=self.max)

my_net = nn.Sequential(
    nn.Linear(7, 5),
    MyActivationFunction()
)

y = my_net(Variable(torch.rand(10, 7)))
y.backward(torch.rand(10, 5))
