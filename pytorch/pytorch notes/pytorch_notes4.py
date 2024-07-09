import torch 
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
print(torch.__version__)
# matrix multiplications
# ----------------------------------------------------------------------------------------------------------
tensor_A = torch.tensor([[1, 2],
                        [3, 4],
                        [5, 6]])

tensor_B = torch.tensor ([[7, 8],
                        [9, 10],
                        [11, 12]])

print(torch.matmul(tensor_A.T, tensor_B))
#tensor aggregation
x = torch.arange(0, 100, 10)
print(torch.min(x))
print(x.max())
print(x.type(torch.float32).mean())