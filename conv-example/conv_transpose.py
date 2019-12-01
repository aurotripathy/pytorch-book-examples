# https://medium.com/@santi.pdp/how-pytorch-transposed-convs1d-work-a7adac63c4a5
# https://arxiv.org/abs/1603.07285 Guide to conv arithmetic
import torch
import torch.nn as nn
import numpy as np

input = torch.ones(1, 1, 3, 3)

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
                 stride=1, padding=0, bias=False)
params = conv.state_dict()
print('params.shape', params['weight'].shape)

params['weight'][0, 0, :, :] = torch.from_numpy(np.ones((3,3)))
conv.load_state_dict(params)

y = conv(input)
print('y:', y)
print('y.shape:', y.shape)

conv_transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3,
                                    stride=1, padding=0, bias=False)
params = conv_transpose.state_dict()
print('params.shape', params['weight'].shape)

params['weight'][0, 0, :, :] = torch.from_numpy(np.ones((3,3)))
conv_transpose.load_state_dict(params)
x = conv_transpose(y)
print('x:', x)
print('x.shape:', x.shape)


input = torch.ones(1, 1, 5, 5)
conv = nn.Conv2d(in_channels=1, out_channels=1,
                 kernel_size=3, stride=1, padding=0, bias=False)
params = conv.state_dict()
print('params.shape', params['weight'].shape)

w = np.ones((3,3)) * 5
w[0, 0] = 4
# w[1, 1] = 4
print('w:', w)

params['weight'][0, 0, :, :] = torch.from_numpy(w)
conv.load_state_dict(params)

y = conv(input)
print('y:', y)
print('y.shape:', y.shape)


conv_transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                 kernel_size=3, stride=3, padding=0, bias=False)
params = conv_transpose.state_dict()
print('params.shape', params['weight'].shape)

params['weight'][0, 0, :, :] = torch.from_numpy(np.ones((3,3)))
conv_transpose.load_state_dict(params)
x = conv_transpose(y)
print('x:', x)
print('x.shape:', x.shape)
