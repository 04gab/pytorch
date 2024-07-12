import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
# creating tensors
#---------------------------------------------------------------------------------------------------------------
tensor = torch.rand(2, 1, 3, 4)
# print(tensor)

image_size_tensor = torch.rand(size= (224, 224, 3)) #height, width, color RBG
print(image_size_tensor.shape)

zeros = torch.zeros(3, 4)
print (zeros*tensor)

one_to_ten = torch.arange(1, 11, 2)
print(one_to_ten)

ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)
#float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                                dtype = None, #datatype of tensor
                                device = None, #device tensor on (GPU, CPU)
                                requires_grad=False)
print(float_32_tensor)
float_16_tensor = float_32_tensor.type(torch.flaot16) #converts the data type tensor