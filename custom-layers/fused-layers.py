import torch
import torch.nn as nn
import numpy as np
import math

def swish(x):
    return x * torch.sigmoid(x)

def fast_gelu(x):
    pi = 3.1415926535897932
    cdf = 0.5 * (1.0 + torch.tanh((math.sqrt(2 / pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x*cdf


activation_str_to_func = {"gelu": fast_gelu,
                          "relu": torch.nn.functional.relu,
                          "swish": swish}

class LinearPlusActivation(torch.nn.Linear):
    """ Fused Linear and activation Module. """

    def __init__(self, in_features, out_features, activation_str='relu', bias=True):
        super().__init__(in_features, out_features, bias)
        if activation_str in activation_str_to_func:
            self.activation_func = activation_str_to_func[activation_str]
        else:
            # raise Exception('Unkown activation string')
            self.activation_func = nn.Identity()


    def forward(self, x):
        return self.activation_func(nn.functional.linear(x, self.weight, self.bias))

linear_plus_gelu = LinearPlusActivation(4, 3, 'gelu')
inp = np.array([[[[1, 2, 3, 4],  # batch(=1) x channels(=1) x height(=3) x width(=4)
                  [1, 2, 3, 4],
                  [1, 2, 3, 4]]]])
x = torch.tensor(inp, dtype=torch.float)
print(linear_plus_gelu(x))
# from ~/nvidia-mlperf/training_results_v0.7/NVIDIA/benchmarks/bert/implementations/pytorch$ emacs modeling.py
# to demo FusedLinearPlusActivation
