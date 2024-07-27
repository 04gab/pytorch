#%%
import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
# tensors
#---------------------------------------------------------------------------------------------------------------
#1 scalar
scalar = torch.tensor(7)
print(scalar.ndim)

vector = torch.tensor([7, 7])
print(vector.ndim)
print(vector.shape)

matrix = torch.tensor([[7,8],
                       [9, 10]])

print(matrix.ndim)
print(matrix.shape)

tensor = torch.tensor([[[1, 2, 3],
                        [3, 6, 9]]])
print(tensor.shape)

# %%
