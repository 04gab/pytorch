#%%
import torch 
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
# Summary
# ----------------------------------------------------------------------------------------------------------
#1.) create device-agnostic code: check of GPU is available, else default CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#%%
weight = 0.7
bias = 0.3
#create range values
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y= weight * X + bias
#%%
#split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
# print(len(X_train), len(y_train), len(X_test), len(y_test))
#%%
#plot data
def plot_predictions(train_data = X_train,
                      train_labels = y_train,
                      test_data = X_test,
                      test_labels = y_test,
                      predictions = None):
    plt.figure(figsize=(10, 7))
    #plot training data in blue
    plt.scatter(train_data, train_labels, c = "b", s = 4, label = "Training data")

    #plot test data in green
    plt.scatter(test_data, test_labels, c = "g", s = 4, label = "Testing data")

    #are there predictions?
    if predictions is not None:
        #plot predictions if they exist
        plt.scatter(test_data, predictions, c = "r", s = 4, label = "Predictions")

        #show legend
    plt.legend(prop={"size": 14})
#%%   
plot_predictions(X_train, y_train, X_test, y_test)
# %%
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features = 1,
                                    out_features=1)