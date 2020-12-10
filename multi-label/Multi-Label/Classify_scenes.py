import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import json
from torch.utils.data import Dataset, DataLoader ,random_split
from PIL import Image
from pathlib import Path
from scipy.io import loadmat
import os, copy


classLabels = ["desert", "mountains", "sea", "sunset", "trees" ]
print(torch.__version__)

df = pd.DataFrame({"image": sorted([ int(x.name.strip(".jpg")) for x in Path("original").iterdir()])})
df.image = df.image.astype(np.str)
print(df.dtypes)
df.image = df.image.str.cat([".jpg"]*len(df))


for label in classLabels:
  df[label]=0
with open("labels.json") as infile:
    s ="["
    s = s + ",".join(infile.readlines())
    s = s + "]"
    s = np.array(eval(s))
    s[s<0] = 0
    df.iloc[:, 1:] = s
df.to_csv("data.csv", index=False)
print(df.head(10))

df = pd.read_csv("data.csv")
fig1, ax1 = plt.subplots()
df.iloc[:,1:].sum(axis=0).plot.pie(autopct='%1.1f%%',shadow=True, startangle=90,ax=ax1)
ax1.axis("equal")
# plt.show()


def visualizeImage(idx):
  fd = df.iloc[idx]
  image = fd.image
  label = fd[1:].tolist()
  print(image)
  image = Image.open("original/"+image)
  fig,ax = plt.subplots()
  ax.imshow(image)
  ax.grid(False)
  classes =  np.array(classLabels)[np.array(label,dtype=np.bool)]
  for i , s in enumerate(classes):
    ax.text(0 , i*20  , s , verticalalignment='top', color="white", fontsize=16, weight='bold')
  plt.show()

visualizeImage(52)

class MyDataset(Dataset):
  def __init__(self , csv_file , img_dir , transforms=None ):
    
    self.df = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.transforms = transforms
    
  def __getitem__(self,idx):
    # d = self.df.iloc[idx.item()]
    d = self.df.iloc[idx]
    image = Image.open(self.img_dir/d.image).convert("RGB")
    label = torch.tensor(d[1:].tolist() , dtype=torch.float32)
    
    if self.transforms is not None:
      image = self.transforms(image)
    return image,label
  
  def __len__(self):
    return len(self.df)

batch_size=32
transform = transforms.Compose([transforms.Resize((224,224)) , 
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

dataset = MyDataset("data.csv" , Path("original") , transform)
valid_no = int(len(dataset)*0.12) 
trainset ,valset  = random_split( dataset , [len(dataset) -valid_no  ,valid_no])
print(f"trainset len {len(trainset)} valset len {len(valset)}")
dataloader = {"train":DataLoader(trainset , shuffle=True , batch_size=batch_size),
              "val": DataLoader(valset , shuffle=True , batch_size=batch_size)}


def create_head(num_features, number_classes, dropout_prob=0.3, activation_func=nn.ReLU):
  features_lst = [num_features , num_features//2 , num_features//4]
  layers = []
  for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
    layers.append(nn.Linear(in_f , out_f))
    layers.append(activation_func())
    layers.append(nn.BatchNorm1d(out_f))
    if dropout_prob !=0 : layers.append(nn.Dropout(dropout_prob))
  layers.append(nn.Linear(features_lst[-1] , number_classes))
  return nn.Sequential(*layers)

def create_model_plus_head():
  model = models.resnet50(pretrained=True) # load the pretrained model
  num_features = model.fc.in_features # get the no of on_features in last Linear unit
  print(num_features)
  ## freeze the entire convolution base
  for param in model.parameters():
    param.requires_grad_(False)
  top_head = create_head(num_features , len(classLabels)) # because ten classes
  model.fc = top_head # replace the fully connected layer
  print(model)
  return model


class ExtendedModel(nn.Module):
  def __init__(self, number_classes, dropout_prob=0.3, activation_func=nn.ReLU):
    super().__init__()
    self.rn50 = models.resnet50(pretrained=True) # load the pretrained model
    num_features = self.rn50.fc.in_features # get the no of on_features in last Linear unit
    for param in self.rn50.parameters():
      param.requires_grad_(False)
    self.features_lst = [num_features , num_features//2 , num_features//4]
    self.dropout_prob=dropout_prob
    self.activation_func = activation_func
    self.number_classes = number_classes

    layers = []
    for in_f, out_f in zip(self.features_lst[:-1] , self.features_lst[1:]):
      layers.append(nn.Linear(in_f , out_f))
      layers.append(activation_func())
      layers.append(nn.BatchNorm1d(out_f))
      if self.dropout_prob !=0 : layers.append(nn.Dropout(self.dropout_prob))
    layers.append(nn.Linear(self.features_lst[-1] , self.number_classes))
    top_head = nn.Sequential(*layers)
    self.rn50.fc = top_head
    print(self.rn50
    )

  def forward(self, x):
    x = self.rn50(x)
    return x

torch.manual_seed(0)
model = ExtendedModel(number_classes=5)
# model = create_model_plus_head()

import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005 )


from tqdm import trange
from sklearn.metrics import precision_score,f1_score

def train(model , data_loader , criterion , optimizer ,scheduler, num_epochs=5):

  for epoch in trange(num_epochs,desc="Epochs"):
    result = []
    for phase in ['train', 'val']:
      if phase=="train":     # put the model in training mode
        model.train()
        scheduler.step()
      else:     # put the model in validation mode
        model.eval()
       
      # keep track of training and validation loss
      running_loss = 0.0
      running_corrects = 0.0  
      
      for data , target in data_loader[phase]:
        #load the data and target to respective device
        data , target = data.to(device)  , target.to(device)

        with torch.set_grad_enabled(phase=="train"):
          #feed the input
          output = model(data)
          #calculate the loss
          loss = criterion(output,target)
          preds = torch.sigmoid(output).data > 0.5
          preds = preds.to(torch.float32)
          
          if phase=="train"  :
            # backward pass: compute gradient of the loss with respect to model parameters 
            loss.backward()
            # update the model parameters
            optimizer.step()
            # zero the grad to stop it from accumulating
            optimizer.zero_grad()


        # statistics
        running_loss += loss.item() * data.size(0)
        running_corrects += f1_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="samples")  * data.size(0)
        
        
      epoch_loss = running_loss / len(data_loader[phase].dataset)
      epoch_acc = running_corrects / len(data_loader[phase].dataset)

      result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    print(result)

train(model,dataloader , criterion, optimizer,sgdr_partial,num_epochs=10)