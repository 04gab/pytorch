#%%
import torch 
from torch import nn
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
#building model (classify blue and red dots)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
# %%
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, #takes in 2 features and upscales to 5 features
                                 out_features=5)
        self.layer_2 = nn.Linear(in_features=5, #takes in 5 features from previous layer and outputs 1 feature (same shape as y)
                                 out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x)) #x -> layer_1 -> layer_2 -> output
    
#instantiate an instance of model and send to target device
model_0 = CircleModel().to(device)
next(model_0.parameters()).device

# %%
#replicated model
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
model_0
# %%
model_0.state_dict()
# %%
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\nFirst 10 predictions: \n {untrained_preds[:10]}")
print(f"\nFirst 10 Labels: \n{y_test[:10]}")
# %%
#loss_fn = nn.BCELoss() #requires input to have gone through the sigmoid activation function prior to input to BCELoss
loss_fn = nn.BCEWithLogitsLoss() #sigmoid activation function built in

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)
#calculate accuracy - out of 100 examples, what # does our model get right?

def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) *100
    return acc
#%%
#model's output are raw "logits" -> convert raw logits into prediction probabilities by passing them to activation function 

#view first 5 outputs of forawrd pass on the test data
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
# %%
#use sigmoid activation function on our model logits
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
# %%
torch.round(y_pred_probs)
#%%
y_preds = torch.round(y_pred_probs)
#logits -> pred probs -> pred labels
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
y_preds.squeeze()
# %%
y_test[:5]
# %%
