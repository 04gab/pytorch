#%%
import torch 
from torch import nn
import torch.utils
import torch.utils.data
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import requests
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from helper_functions import accuracy_fn
print(torch.__version__)
print(torchvision.__version__)
# computer vision
# ----------------------------------------------------------------------------------------------------------
# %%
train_data = datasets.FashionMNIST( #60,000 samples
    root="data", #where to download data to
    train=True, #use training set
    download=True, 
    transform=torchvision.transforms.ToTensor(), #how do we want to transform data
    target_transform=None #how do we want to transform labels/targets
)

test_data = datasets.FashionMNIST( #10,00 samples
    root="data", 
    train=False, 
    download=True, 
    transform=ToTensor(), 
    target_transform=None 
)
# %%
len(train_data), len(test_data)
# %%
image, label = train_data[0]
# %%
class_names = train_data.classes
class_names
# %%
class_to_idx = train_data.class_to_idx
class_to_idx
# %%
print(f"Image shape: {image.shape} -> [color channels, width, height]")
print(f"Image Label: {class_names[label]}")
# %%
#Visualizing data
plt.imshow(image.squeeze()) #squeeze the dim 
plt.title(label)
# %%
plt.imshow(image.squeeze(), cmap="grey")
plt.title(class_names[label])
plt.axis(False)
# %%
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="grey")
    plt.title(class_names[label])
    plt.axis(False)
# %%
train_data, test_data
# %%
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
# %%
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Lentgh of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}...")
#%%
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape
# %%
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="grey")
plt.title(class_names[label])
plt.axis(False)
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
# %%
flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)

print(f"Shape before flattening: {x.shape}")
print(f"Shape after flattening: {output.shape}")

# %%
class FashinMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, 
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units, 
                      out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

#%%
model_0 = FashinMNISTModelV0(
    input_shape=784, #28*28
    hidden_units=10,
    output_shape=len(class_names)
).to("cpu")
model_0

#%%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# %%
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
   total_time = end - start
   print(f"Train time on {device}: {total_time:.3f} seconds")
   return total_time


# %%
torch.manual_seed(42)
train_time_start_on_cpu = timer() 

# Set the number of epochs (we'll keep this small for faster training time)
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss

        optimizer.zero_grad()

        loss.backward()

        # 5. Optimizer step (update the model's parameters once *per batch*)
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

        train_loss /= len(train_dataloader)

        test_loss, test_acc = 0, 0
        model_0.eval()
    with torch.inference_mode(): 
        for X_test, y_test in test_dataloader:
            test_pred = model_0(X_test)

            test_loss += loss_fn(test_pred, y_test)

            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(test_dataloader)

            test_acc /= len(test_dataloader)

    print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))
#%%
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, 
               accuracy_fn):
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in tqdm(data_loader):
      y_pred = model(X)

      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true=y,
                         y_pred=y_pred.argmax(dim=1))

    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name": model.__class__.__name__, # only works when model was created with a class
          "model_loss": loss.item(),
          "model_acc": acc}

model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn, 
                             accuracy_fn=accuracy_fn)
model_0_results
#%%
