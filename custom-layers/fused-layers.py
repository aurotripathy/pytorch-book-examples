import torch
import torch.nn as nn

class LinearPlusActivation(nn.Linear):
    """ Fused Linear and activation Module. """

    def __init__(self, in_features, out_features, activation_str='relu', bias=True):
        super().__init__(in_features, out_features, bias)
        if activation_str in activation_str_to_func:
            self.activation_func = activation_str_to_func[activation_str]
        else:
            raise Exception('Unkown activation string')

    def forward(self, x):
        return self.activation_func(nn.functional.linear(x, self.weight, self.bias))


activation_str_to_func = {"gelu": nn.functional.gelu,
                          "relu": nn.functional.relu,}

torch.manual_seed(0)  # for repeatable results    
linear_plus_gelu = LinearPlusActivation(4, 2, 'gelu')
x = torch.tensor([[[[1, 2, 3, 4],  # batch(=1) x channels(=1) x height(=3) x width(=4)
                    [5, 6, 7, 8],
                    [9, 1, 2, 3]]]], dtype=torch.float32)
print(linear_plus_gelu(x))  # fused layer

# Unfused version that gives the same output
torch.manual_seed(0)  # reset the random num generator
linear = nn.Linear(4, 2, True)
gelu_activation = nn.GELU()
print(gelu_activation(linear(x)))
