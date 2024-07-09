import torch 
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
print(torch.__version__)
# tensors operations
#---------------------------------------------------------------------------------------------------------------
tensor = torch.tensor([1, 2, 3])
tensor = tensor + 10
print(tensor)
tensor = tensor * 10
print(tensor)
tensor = tensor - 10
print(tensor)
tensor = tensor % 10
print(tensor)
mat_tensor = torch.tensor([1, 2, 3])
start = timeit.default_timer()
print(torch.matmul(mat_tensor, mat_tensor))
stop = timeit.default_timer()

print('Time: ', stop - start)  




