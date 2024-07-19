#%%
import torch 
import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
print(torch.__version__)
# classification
# ----------------------------------------------------------------------------------------------------------
#%%
#data set, make 1000 samples
n_samples = 1000
#create circles
X, y = make_circles(n_samples,
                    noise = 0.03,
                    random_state=42) #seed

print(len(X), len(y))
# %%
print(f"First 5 samples of X: \n {X[:5]}")

print(f"First 5 samples of y: \n {y[:5]}")
# %%
#make DataFrame of cirlce data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
circles.head(10)
# %%
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
# %%
print(X.shape, y.shape)
# %%
X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shape for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

# %%
#turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y=  torch.from_numpy(y).type(torch.float)

print(X[:5], y[:5])
# %%
torch.manual_seed(42)
#%%
#split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,#20% of data will be test 80% will be train
                                                    random_state=42) 
# %%
print(len(X_train), len(X_test), len(y_train), len(y_test))
# %%
