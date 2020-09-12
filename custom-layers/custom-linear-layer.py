""" Demonstrates the easy of integration of a custom layer """
import math
import torch
import torch.nn as nn
import numpy as np

class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(torch.Tensor(size_out, size_in))  # nn.Parameter is a Tensor that's a module parameter.
        self.bias = nn.Parameter(torch.Tensor(size_out))

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):  # w times x + b
        return (torch.mm(x, self.weights.t())) + self.bias


class BasicModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.conv = nn.Conv2d(1, 128, 3)
        if mode == 'use-library':
            self.linear = nn.Linear(256, 2)
        elif mode == 'use-custom':
            self.linear = MyLinearLayer(256, 2)
        else:
            raise Exception('Sorry, must be either use-library or use-custom')

    def forward(self, x):
        x = self. conv(x)
        x = x.view(-1, 256)
        return self.linear(x)

    
torch.manual_seed(0)  #  for repeatable results
inp = np.array([[[[1, 2, 3, 4],  # batch(=1) x channels(=1) x height x width
                  [1, 2, 3, 4],
                  [1, 2, 3, 4]]]])
x = torch.tensor(inp, dtype=torch.float)

basic_model = BasicModel('use-library')
print('Forward computation by using torch library:', basic_model(x))

torch.manual_seed(0)  #  important to reset 
basic_model = BasicModel('use-custom')
print('Forward computation by using custom layer :', basic_model(x))
