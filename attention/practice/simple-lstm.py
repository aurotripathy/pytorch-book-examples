# for vscode use ctl+shift+P and select the right python interpreter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optin
# from pudb import set_trace


torch.manual_seed(1)

# set_trace()
lstm = nn.LSTM(3, 3)  # in dim 3, put dim 3
inputs = [torch.randn(1, 3) for _ in range(5)]

# initialize the hidden state

hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)


# Altermately
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print(inputs)
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

