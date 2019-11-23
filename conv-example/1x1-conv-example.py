""" 
1x1 convolution
"""
import torch
import numpy as np

# three channels (or feature maps)
# 4D matrix, first dimension represents batch-size
input = torch.zeros([1, 3, 5, 5], dtype=torch.float32)
# Init inputs with padding all around
i0 = np.array([[1, 2, 1, 0, 2],
               [2, 2, 1, 0, 0],
               [2, 0, 2, 2, 0],
               [2, 1, 2, 2, 2],
               [1, 2, 0, 2, 0]])

input[0, 0, :, :] = torch.from_numpy(i0)

i1 = np.array([[2, 1, 2, 2, 1],
               [2, 1, 0, 1, 0],
               [1, 0, 0, 0, 2],
               [2, 2, 2, 0, 1],
               [1, 0, 1, 2, 2]])

input[0, 1, :, :] = torch.from_numpy(i1)

i2 = np.array([[2, 1, 1, 0, 1],
               [2, 2, 1, 0, 2],
               [0, 0, 0, 1, 2],
               [2, 2, 1, 1, 0],
               [1, 2, 1, 0, 2]])
input[0, 2, :, :] = torch.from_numpy(i2)


print('Input shape:\n', input.shape)
print('Input:\n', input)

class OneByOneConvIn3Out2(torch.nn.Module):
    """ 2D convolution with three inputs, two outputs """ 
    def __init__(self, input_size, output_size):
        super(OneByOneConvIn3Out2, self).__init__()
        kernel_size = 1  # 1 by 1
        self.conv = torch.nn.Conv2d(input_size,
                                    output_size,
                                    kernel_size,
                                    stride=1,
                                    padding=0,
                                    bias=False)
        
    def forward(self, x):
        out = self.conv(x)
        return out

convolve = OneByOneConvIn3Out2(3,2) # Instantiate

# Init w/known weights, replacing the random weights
w00 = np.array([[1]])
w01 = np.array([[2]])
w02 = np.array([[1]])

w10 = np.array([[1]])
w11 = np.array([[3]])
w12 = np.array([[1]])


params = convolve.state_dict()
print('Parameter weights shape:\n', params['conv.weight'].shape)
print('Reinitialize params')

params['conv.weight'][0, 0, :, :] = torch.from_numpy(w00)
params['conv.weight'][0, 1, :, :] = torch.from_numpy(w01)
params['conv.weight'][0, 2, :, :] = torch.from_numpy(w02)

params['conv.weight'][1, 0, :, :] = torch.from_numpy(w10)
params['conv.weight'][1, 1, :, :] = torch.from_numpy(w11)
params['conv.weight'][1, 2, :, :] = torch.from_numpy(w12)

# Loading the state dict is absolutely necessary
convolve.load_state_dict(params)

print('Initialized state dict(weights and bias):\n',
      convolve.state_dict())

# Apply the forward pass
output = convolve(input)
print('Output shape:\n', output.shape)
print('Output of 1x1 covolution:\n', output)
