import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
# GPU acceleration
# ----------------------------------------------------------------------------------------------------------
print(torch.cuda.is_available())
#setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
#count # of devices
print(torch.cuda.device_count())
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu.device)
#numpy does operations only on CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)