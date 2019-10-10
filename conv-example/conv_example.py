""" 
Show the workings of a convolution with PyTorch.
Data from http://cs231n.github.io/convolutional-networks/
Results consistent with the results on the web-site
"""
import torch
import numpy as np

# three channels (or feature maps)
# 4D matrix, one dimension represents batch-size
input = torch.zeros([1, 3, 7, 7], dtype=torch.float32)
# Init inputs with padding all around
i0 = np.array([[0,0,0,0,0,0,0],
               [0,1,2,1,0,2,0],
               [0,2,2,1,0,0,0],
               [0,2,0,2,2,0,0],
               [0,2,1,2,2,2,0],
               [0,1,2,0,2,0,0],
               [0,0,0,0,0,0,0]])
input[0, 0, :, :] = torch.from_numpy(i0)

i1 = np.array([[0,0,0,0,0,0,0],
               [0,2,1,2,2,1,0],
               [0,2,1,0,1,0,0],
               [0,1,0,0,0,2,0],
               [0,2,2,2,0,1,0],
               [0,1,0,1,2,2,0],
               [0,0,0,0,0,0,0]])
input[0, 1, :, :] = torch.from_numpy(i1)

i2 = np.array([[0,0,0,0,0,0,0],
               [0,2,1,1,0,1,0],
               [0,2,2,1,0,2,0],
               [0,0,0,0,1,2,0],
               [0,2,2,1,1,0,0],
               [0,1,2,1,0,2,0],
               [0,0,0,0,0,0,0]])
input[0, 2, :, :] = torch.from_numpy(i2)


print('Input shape:\n', input.shape)
print('Input:\n', input)

class ConvIn3Out2(torch.nn.Module):
    """ 2D convolution with three inputs, two outputs """ 
    def __init__(self, input_size, output_size):
        super(ConvIn3Out2, self).__init__()
        kernel_size = 3  # 3 by 3
        self.conv = torch.nn.Conv2d(input_size,
                                    output_size,
                                    kernel_size,
                                    stride=2,
                                    padding=0,
                                    bias=True)
        
    def forward(self, x):
        out = self.conv(x)
        return out

convolve = ConvIn3Out2(3,2) # Instantiate

# Init the weights, replacing the random weights
w00 = np.array([[1, 0, -1],
                [-1, 1, 0],
                [0, 1, 0]])

w01 = np.array([[-1, 1, 1],
                [1, 1, -1],
                [1, 0, 1]])

w02 = np.array([[0, 0, 1],
                [1, 1, -1],
                [0, 0, -1]])
# --

w10 = np.array([[1, -1, 1],
                [0, -1, 0],
                [0, -1, -1]])

w11 = np.array([[-1, 0, 0],
                [0, 0, 0],
                [1, 0, 0]])

w12 = np.array([[-1, -1, -1],
                [-1, -1, 1],
                [0, 1, -1]])

params = convolve.state_dict()
print('Parameter weights shape:\n', params['conv.weight'].shape)
params['conv.weight'][0, 0, :, :] = torch.from_numpy(w00)
params['conv.weight'][0, 1, :, :] = torch.from_numpy(w01)
params['conv.weight'][0, 2, :, :] = torch.from_numpy(w02)

params['conv.weight'][1, 0, :, :] = torch.from_numpy(w10)
params['conv.weight'][1, 1, :, :] = torch.from_numpy(w11)
params['conv.weight'][1, 2, :, :] = torch.from_numpy(w12)

# Init bias
print('Parameter bias shape:\n', params['conv.bias'].shape)
params['conv.bias'] = torch.tensor([1., 0.])


# Loading the state dict is absolutely necessary
convolve.load_state_dict(params)
print('Initialized state dict(weights and bias):\n',
      convolve.state_dict())

# Apply the forward pass
result = convolve(input)
print('Output of covolution:\n', result)
