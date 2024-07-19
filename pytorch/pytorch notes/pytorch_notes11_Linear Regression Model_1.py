#%%
import torch 
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
print(torch.__version__)
# Linear Regression model V2
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
        self.linear_layer = nn.Linear(in_features = 1, #size one in x/y (in_features. out_features) 
                                    out_features=1)
    
    def forward (self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

#set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1, model_1.state_dict())
# %%
#check model current device
print(next(model_1.parameters()).device)
# %%
model_1.to(device)
print(next(model_1.parameters()).device)
# %%
# Training:
# loss functionoptimizer
# training Loop
# testing loop
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)
torch.manual_seed(42)
epochs = 200
#put data on target device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_1.train()
    # forward pass
    y_pred = model_1(X_train)
    #calculate loss
    loss = loss_fn(y_pred, y_train)
    #optimizer zero grad
    optimizer.zero_grad()
    #backpropagation
    loss.backward()
    #optimizer step
    optimizer.step()
    #testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss {test_loss}")

# %%
print(model_1.state_dict())
# %%
#turn model into evaluation mode
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test) 

# x_test is on gpu, covert to cpu to use plot
plot_predictions(predictions=y_preds.cpu())
# %%
#saving & loading model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_LinearRegressionModel_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")

torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)
# %%
#load model
#create a new instantce of linear regression model V2
loaded_model_1 = LinearRegressionModelV2()

loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

#put loaded model to device
loaded_model_1.to(device)
#%%
next(loaded_model_1.parameters()).device
print(loaded_model_1.state_dict())
# %%
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)
y_preds == loaded_model_1_preds

# %%
