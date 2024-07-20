#%%
import torch 
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
print(torch.__version__)
# non linear activation functions
# ----------------------------------------------------------------------------------------------------------
#%%
A = torch.arange(-10, 10, 1, dtype=torch.float32)
plt.plot(torch.relu(A))
# %%
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x) #inputs must be tensors
relu(A)
# %%
plt.plot(relu(A))
# %%
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
#%%
plt.plot(torch.sigmoid(A))
# %%
plt.plot(sigmoid(A))
# %%
