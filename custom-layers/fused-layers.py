import torch
import torch.nn as nn
import numpy as np
import math

def swish(x):
    """ Swish activation works better than ReLU on deeper models across a number of challenging data sets. """
    return x * torch.sigmoid(x)

def gelu(x):
    """ Gaussian Error Linear Unit, defined as the input (x) times 
        standard Gaussion cumulative dustribution function. """
    pi = 3.1415926535897932
    cdf = 0.5 * (1.0 + torch.tanh((math.sqrt(2 / pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x*cdf


class LinearPlusActivation(torch.nn.Linear):
    """ Fused Linear and activation Module. """

    def __init__(self, in_features, out_features, activation_str='relu', bias=True):
        super().__init__(in_features, out_features, bias)
        if activation_str in activation_str_to_func:
            self.activation_func = activation_str_to_func[activation_str]
        else:
            raise Exception('Unkown activation string')

    def forward(self, x):
        return self.activation_func(nn.functional.linear(x, self.weight, self.bias))


activation_str_to_func = {"gelu": gelu,
                          "relu": torch.nn.functional.relu,
                          "swish": swish}

torch.manual_seed(0)  # for repeatable results    
linear_plus_gelu = LinearPlusActivation(4, 2, 'gelu')
inp = np.array([[[[1, 2, 3, 4],  # batch(=1) x channels(=1) x height(=3) x width(=4)
                  [1, 2, 3, 4],
                  [1, 2, 3, 4]]]])
x = torch.tensor(inp, dtype=torch.float)
print(linear_plus_gelu(x))  # fused layer

torch.manual_seed(0)  # reset the random num generator
linear = torch.nn.Linear(4, 2)
gelu_activation = nn.GELU()
print(gelu_activation(linear(x)))  # unfused


