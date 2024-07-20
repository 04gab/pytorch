#%%
import torch 
from torch import nn
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
import requests
from pathlib import Path
print(torch.__version__)
# troubleshooting large problem by testing smaller problem
# ----------------------------------------------------------------------------------------------------------
#%%
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary
#%%
weight =  0.7
bias = 0.3
start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

print(len(X_regression))
X_regression[:5], y_regression[:5]
# %%
train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]
len(X_train_regression), len(X_test_regression), len(y_train_regression), len(y_test_regression)

# %%
plot_predictions(train_data=X_train_regression,
                  train_labels=y_train_regression,
                  test_data=X_test_regression,
                  test_labels=y_test_regression)
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

model_2 = nn.Sequential(
  nn.Linear(in_features=1, out_features=10),
  nn.Linear(in_features=10, out_features=10),
  nn.Linear(in_features=10, out_features=1)
).to(device)
model_2
# %%
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr= 0.01)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

for epoch in range(epochs):
  y_pred = model_2(X_train_regression)
  loss = loss_fn(y_pred, y_train_regression)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model_2.eval()
  with torch.inference_mode():
    test_pred = model_2(X_test_regression)
    test_loss = loss_fn(test_pred, y_test_regression)

    if epoch % 100 == 0:
      print(f"Epoch: { epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")
# %%

model_2.eval()
with torch.inference_mode():
  y_preds = model_2(X_test_regression)

plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu())
# %%
