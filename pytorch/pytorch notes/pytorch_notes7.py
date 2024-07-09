import torch 
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
print(torch.__version__)
# pytorch and numpy
# ----------------------------------------------------------------------------------------------------------
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)
array = array + 1
print(array, tensor)