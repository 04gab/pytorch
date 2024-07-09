import torch 
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
print(torch.__version__)
# reshpae view, squeeze, unsqueeze
# ----------------------------------------------------------------------------------------------------------
x = torch.arange(1., 11.)
print(x)
x_reshaped = x.reshape(2, 5)
print(x_reshaped)
#view shares the same memory as original tensor
z= x.view(1, 10)
z[:, 0] = 5
print(z, x)
#stack tensors on top of each other
x_stacked = torch.stack([x, x, x,], dim=1) # 0 = vertical, 1 = horizontial
print(x_stacked)
#squeeze -> removes all single dimension from tensor
print(x_reshaped.squeeze())
print(x_reshaped.squeeze().shape)
#unsqueeze -> adds a single dimension to a tensor at a specitif dim
x_unsqueezed = x.unsqueeze(dim = 0)
print(x.shape)
print(x_unsqueezed)
print(x_unsqueezed.shape)
#permute -> rearrange the dimensions of the tensor 
x_original = torch.rand(size= (224, 224, 3)) #height, width, color channel
#permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1)
print(f"original shape: {x_original.shape}")
print(f"permuted shape: {x_permuted.shape}")
