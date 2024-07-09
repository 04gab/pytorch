#%%
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
print(torch.__version__)
# dataset
# ----------------------------------------------------------------------------------------------------------

weight = 0.7
bias = 0.3
#create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# print(X[:10], y[:10])
#create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(y_train), len(X_test), len(y_test))

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
plot_predictions()


# %%
#build linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                 requires_grad = True,
                                                 dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad = True,
                                             dtype=torch.float))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

#%%
#manuel seed
torch.manual_seed(42)

model_0 = LinearRegressionModel()

#check out the parameters
print(list(model_0.parameters()))
#check out our model's parameters (a parameter is a value that the model sets itself)
print(model_0.state_dict())

#%%
y_preds = model_0(X_test)
print(y_preds)
# %%
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)
#%%
plot_predictions(predictions=y_preds)

# %%
#loss function
loss_fn = nn.L1Loss()

#optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.01) #lr = learning rate (most important hpyerparameter you can set)

#an eoich is on loop through the data (hpyerparameter)
#%%
torch.manual_seed(42)
epochs = 200
#track values
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    #set the model to training mode
    model_0.train() # train mode in PyTorch sets all parameters that require gradients to requires gradients
    
    #forward pass
    y_pred = model_0(X_train)

    #calculate the loss
    loss = loss_fn(y_pred, y_train)
    # print(f"Loss: {loss}")

    #optimizer zero grad (they accumulate every epoch by default)
    optimizer.zero_grad()
    
    #perform back prop on the loss with respect to the parameters of the model
    loss.backward()

    #step the optimizer (perform gradient descent)
    optimizer.step() #by default how the optimizer changes will acculumate through the loop

    model_0.eval() #turns off settings not needed for evaluation/testing (dropout/batch norm layers)
    with torch.inference_mode(): #turns off gradient tracking
        #forward pass
        test_pred = model_0(X_test)
        #calculate the loss
        test_loss = loss_fn(test_pred, y_test)
    
    #print whats happening
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
        print(model_0.state_dict())
#%%
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label = "Train loss")
plt.plot(epoch_count, test_loss_values, label = "Test loss")
plt.title("Training and test loss curvers")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
#%%
with torch.inference_mode():
    y_preds_new = model_0(X_test)
print(model_0.state_dict())

#%%
print(weight, bias)
#%%
plot_predictions(predictions=y_preds)
#%%
plot_predictions(predictions=y_preds_new)

# %%
#Saving pytorch model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#create model save path
MODEL_NAME = "01_LinearRegressionModel.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

# %%
#loading pytorch model
#to load in a saved state_dict you have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel()

#load the saved sate dict of model_0 (this will update the new instance with updated parameters)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model_0.state_dict())

#%%
#make some predictions with our loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)
print(loaded_model_preds)
# %%
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)
#%%
y_preds == loaded_model_preds
# %%
