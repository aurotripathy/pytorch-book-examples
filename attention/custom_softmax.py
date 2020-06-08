

# 163def softmax(x, axis=1):                                                                       
# 164    """Softmax activation function.                                                           
# 165    # Arguments                                                                               
# 166        x : Tensor.                                                                           
# 167        axis: Integer, axis along which the softmax normalization is applied.                 
# 168    # Returns                                                                                 
# 169        Tensor, output of softmax transformation.                                             
# 170    # Raises                                                                                  
# 171        ValueError: In case `dim(x) == 1`.                                                    
# 172    """                                                                                       
# 173    ndim = K.ndim(x)                                                                          
# 174    if ndim == 2:                                                                             
# 175        return K.softmax(x)                                                                   
# 176    elif ndim > 2:                                                                            
# 177        e = K.exp(x - K.max(x, axis=axis, keepdims=True))                                     
# 178        s = K.sum(e, axis=axis, keepdims=True)                                                
# 179        return e / s                                                                          
# 180    else:                                                                                     
# 181        raise ValueError('Cannot apply softmax to a tensor that is 1D')         



# https://discuss.pytorch.org/t/customize-an-activation-function/1652/6
import torch
import torch.nn as nn
import torch.nn.functional as F
from pudb import set_trace
import numpy as np

class SoftMax(nn.Module):
    """Softmax activation function.                                                           
    # Arguments                                                                               
        x : Tensor.                                                                           
        axis: Integer, axis along which the softmax normalization is applied.                 
    # Returns                                                                                 
        Tensor, output of softmax transformation.                                             
    # Raises                                                                                  
        ValueError: In case `dim(x) == 1`.                                                    
    """                                                                                       

    def __init__(self, axis=1):
        super(SoftMax, self).__init__()
        self.axis = axis

    def forward(self, x):
        return F.softmax(x, dim=self.axis)

def sig(np_data):
    return np.exp(np_data)/np.sum(np.exp(np_data))
    
# set_trace()
data = torch.randn((5,5))
print('tensor data\n', data)
np_data = data.numpy()
print('numpy data\n', np_data)
sig_along_axis =np.apply_along_axis(sig, 1, np_data) 
print('numpy sigmoid along axis 1\n', sig_along_axis)

sig_along_axis =np.apply_along_axis(sig, 0, np_data) 
print('numpy sigmoid along axis 0\n', sig_along_axis)


sm = SoftMax(1)
print('torch.forward\n', sm.forward(data))
print('torch sigmoid thru forward prop\n', sm(data))
print('Done!')
sm = SoftMax(0)
print('torch.forward\n', sm.forward(data))
print('torch sigmoid thru forward prop\n', sm(data))
print('Done!')

