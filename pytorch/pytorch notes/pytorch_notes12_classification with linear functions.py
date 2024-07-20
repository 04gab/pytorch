#%%
import torch 
from torch import nn
import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests
from pathlib import Path
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

def accuracy_fn(y_true, y_pred):
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
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

#build training and evaluation loop
for epoch in range(epochs):
    #trainning
    model_0.train()
    #forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) #turn logits -> pred probs -> pred labels

    #calcualte loss, accuracy
    loss = loss_fn(y_logits, #nn.BCEWithLogitsLoss expects raw logits as input
                    y_train)
    acc = accuracy_fn(y_true=y_train,
                   y_pred=y_pred)
    #optimizer zero grad
    optimizer.zero_grad()

    #loss backward
    loss.backward()

    #optimizer step (gradient desecnt)
    optimizer.step()

    #testing
    model_0.eval()
    with torch.inference_mode():
        #forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        #calculate test loss/acc
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
        if epoch % 10 ==0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% |Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
# %%
#make predictions and evaluate the model

if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary
# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
#-------------------------------------------------------------------------------------------------------
# %%
#improving a model (from a model perspective)
#add more layers - give the model more chances to learn about patterns in the data
#add more hidden units - 5 hidden in_features to 10 
#fit for longer (more epochs)
#Changing activation function
#change learning rate
#change the loss function

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
model_1 = CircleModelV1().to(device)
# %%
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr = 0.1)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):

    model_1.train()
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        loss = loss_fn(test_logits, 
                       y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
# %%
