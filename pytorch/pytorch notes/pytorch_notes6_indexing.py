import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
# indexing
# ----------------------------------------------------------------------------------------------------------
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)
print(f"first index: {x[0]}")
print(f"second index: {x[0][0]}")
print(f"third index: {x[0][0][0]}")
#use ":" to select all of a target dimension
print(x[:, :, 2]) #get all values on 0th and 1st dimenion but only index 2 of 2nd dimension