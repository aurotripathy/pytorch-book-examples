""" 1x1 convolution """
import torch
import numpy as np

class OneByOneConvIn3Out2(torch.nn.Module):
    """ 1x1 convolution with three inputs, two outputs """ 
    def __init__(self, input_size, output_size):
        super(OneByOneConvIn3Out2, self).__init__()
        kernel_size = 1  # 1 by 1
        self.conv = torch.nn.Conv2d(input_size, output_size,
                                    kernel_size, stride=1,
                                    padding=0, bias=False)
        
    def forward(self, x):
        out = self.conv(x)
        return out

# three channels (aka feature maps); 4D tensor, 1st dimension is the batch-size
input = torch.zeros([1, 3, 5, 5], dtype=torch.float32)

# Init inputs 
channel_0 = np.array([[1, 2, 1, 0, 2],
                      [2, 2, 1, 0, 0],
                      [2, 0, 2, 2, 0],
                      [2, 1, 2, 2, 2],
                      [1, 2, 0, 2, 0]])
input[0, 0, :, :] = torch.from_numpy(channel_0)

channel_1 = np.array([[2, 1, 2, 2, 1],
                      [2, 1, 0, 1, 0],
                      [1, 0, 0, 0, 2],
                      [2, 2, 2, 0, 1],
                      [1, 0, 1, 2, 2]])
input[0, 1, :, :] = torch.from_numpy(channel_1)

channel_2 = np.array([[2, 1, 1, 0, 1],
               [2, 2, 1, 0, 2],
               [0, 0, 0, 1, 2],
               [2, 2, 1, 1, 0],
               [1, 2, 1, 0, 2]])
input[0, 2, :, :] = torch.from_numpy(channel_2)

print('Input shape:\n', input.shape)
print('Input:\n', input)

convolve = OneByOneConvIn3Out2(3,2) # Instantiate

# The shape of weights are [2, 3, 1, 1]; 2 x 3 1x1 kernels
weights = np.array([[[[1]], [[2]], [[1]]],
                    [[[1]], [[3]], [[1]]]])

params = convolve.state_dict()

# Reinitialize params
params['conv.weight'] = torch.from_numpy(weights)

# Reloading the state dict is necessary to init w/known weights
convolve.load_state_dict(params)

print('Initialized state dict:\n',
      convolve.state_dict())

# Apply the forward pass and print
output = convolve(input)
print('Output shape:\n', output.shape)
print('Output after 1x1 covolution:\n', output)
