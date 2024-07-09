import torch 
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
print(torch.__version__)
#reproducibility
# ----------------------------------------------------------------------------------------------------------
#to reduce randomness in NN and pytorch
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)
RANDOM_SEED = 42 #the flavour of seed
torch.manual_seed(RANDOM_SEED) #works for one block of code
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)