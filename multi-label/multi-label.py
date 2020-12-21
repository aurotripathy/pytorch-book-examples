# https://medium.com/the-owl/imbalanced-multilabel-image-classification-using-keras-fbd8c60d7a4b
import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil, os, time, random, copy
# import imageio
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, confusion_matrix, average_precision_score
from skimage.transform import rotate, AffineTransform, warp, resize
from skimage import io
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import trange
from sklearn.metrics import precision_score, f1_score

# from pudb import set_trace

# Data download and extraction
#  wget http://www.lamda.nju.edu.cn/files/miml-image-data.rar
#  unrar e content/miml-image-data.rar
#  mkdir content/original_images
#  unrar e original.rar content/original_images
#  unrar e processed.rar

def display_histogram():
    # Histogram for each individual label
    fig, ax = plt.subplots()

    d_heights, d_bins = np.histogram(data_df['desert'], bins=[-0.5, 0.5, 1.5])
    m_heights, m_bins = np.histogram(data_df['mountains'], bins=d_bins)
    s_heights, s_bins = np.histogram(data_df['sea'], bins=m_bins)
    ss_heights, ss_bins = np.histogram(data_df['sunset'], bins=s_bins)
    t_heights, t_bins = np.histogram(data_df['trees'], bins=ss_bins)

    width = (d_bins[1] - d_bins[0])/6.0

    ax.bar(d_bins[:-1]+width, d_heights, width=width, facecolor='cornflowerblue')
    ax.bar(m_bins[:-1]+width*2, m_heights, width=width, facecolor='seagreen')
    ax.bar(s_bins[:-1]+width*3, s_heights, width=width, facecolor='red')
    ax.bar(ss_bins[:-1]+width*4, ss_heights, width=width, facecolor='blue')
    ax.bar(t_bins[:-1]+width*5, t_heights, width=width, facecolor='yellow')

    plt.show()

content_root = '/home/auro/tf-multi-label-example/content/'
    
# Data processing 
proc_mat = loadmat(os.path.join(content_root, 'miml data.mat'))

class_names = []
for c in proc_mat['class_name']:
    class_names.append(c[0][0])

print(class_names)
#output : ['desert', 'mountains', 'sea', 'sunset', 'trees']

labels = copy.deepcopy(proc_mat['targets'].T)
labels[labels==-1] = 0

data_df = pd.DataFrame(columns=["filenames"] + class_names)  # empty dataframe 
filenames = os.listdir(os.path.join(content_root, "original_images/"))
data_df["filenames"] = np.array(sorted(list(map(lambda x:int(x[:-4]),np.array(filenames)))))
data_df['filenames'] = data_df['filenames'].apply(lambda x:os.path.join(content_root, 'original_images/') + str(x) + '.jpg')
data_df[class_names] = np.array(labels)

#Applying Label Powerset Tranformation
data_df['powerlabel'] = data_df.apply(lambda x : 16*x["desert"]+8*x['mountains']+4*x['sea']+2*x['sunset']+1*x['trees'],axis=1)

data_df['powerlabel'].hist(bins=np.unique(data_df['powerlabel']))
print(data_df)
# output
#                              filenames desert mountains sea sunset trees  powerlabel
# 0        content/original_images/1.jpg      1         0   0      0     0          16
# 1        content/original_images/2.jpg      1         0   0      0     0          16
# 2        content/original_images/3.jpg      1         0   0      0     0          16
# 3        content/original_images/4.jpg      1         1   0      0     0          24
# 4        content/original_images/5.jpg      1         0   0      0     0          16
# ...                                ...    ...       ...  ..    ...   ...         ...
# 1995  content/original_images/1996.jpg      0         0   0      0     1           1
# 1996  content/original_images/1997.jpg      0         0   0      0     1           1
# 1997  content/original_images/1998.jpg      0         0   0      0     1           1
# 1998  content/original_images/1999.jpg      0         0   0      0     1           1
# 1999  content/original_images/2000.jpg      0         0   0      0     1           1

# display_histogram()

# Data Splitting
train_df = pd.DataFrame(columns = ["filenames"] + class_names)
val_df = pd.DataFrame(columns = ["filenames"] + class_names)
train_inds, val_inds = train_test_split(np.array(list(range(data_df.shape[0]))),
                                        test_size=0.2,
                                        random_state=7)
train_df = data_df.iloc[train_inds,:].reset_index(drop=True)
val_df = data_df.iloc[val_inds,:].reset_index(drop=True)

# Oversampling
powercount = {}
powerlabels = np.unique(train_df['powerlabel'])
for p in powerlabels:
    powercount[p] = np.count_nonzero(train_df['powerlabel'] == p)

maxcount = np.max(list(powercount.values()))
for p in powerlabels:
    gapnum = maxcount - powercount[p]
    #print(gapnum)
    temp_df = train_df.iloc[np.random.choice(np.where(train_df['powerlabel']==p)[0],size=gapnum)]
    data_df = train_df.append(temp_df,ignore_index=True)
    
train_df = train_df.sample(frac=1).reset_index(drop=True)

# Calculating class weights
positive_weights = {}
negative_weights = {}
for c in class_names:
    positive_weights[c] = train_df.shape[0] / (2 * np.count_nonzero(train_df[c]==1))
    negative_weights[c] = train_df.shape[0] / (2 * np.count_nonzero(train_df[c]==0))
print('positive weights', positive_weights)
print('negative weights', negative_weights)

#outputs:
#{'desert': 1.4270882491741388, 'mountains': 1.2378223495702005, 'sea': 1.4372623574144487, 'sunset': 1.1101321585903083, 'trees': 1.1113561190738699}
#{'desert': 0.7696614914736574, 'mountains': 0.8388349514563107, 'sea': 0.7667342799188641, 'sunset': 0.9097472924187726, 'trees': 0.90892696122633}


# Calculating the Image Dimensions of the first, use that for the rest
# img = imageio.imread(os.path.join(content_root, 'original_images/1.jpg'))
# H, W, _ = img.shape
# print(f'image shape: {H}, {W}')

class MyDataset(Dataset):
  def __init__(self , csv_file , img_dir , transforms=None ):
    
    self.df = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.transforms = transforms
    
  def __getitem__(self,idx):
    d = self.df.iloc[idx.item()]
    image = Image.open(self.img_dir/d.image).convert("RGB")
    label = torch.tensor(d[1:].tolist() , dtype=torch.float32)
    
    if self.transforms is not None:
      image = self.transforms(image)
    return image,label
  
  def __len__(self):
    return len(self.df)

# Custom Loss Function for imbalanced classes
def loss_fn(y_true, y_pred):
    loss = 0
    loss -= (positive_weights['desert'] * y_true[0] * K.log(y_pred[0]) +
             negative_weights['desert'] * (1 - y_true[0]) * K.log(1 - y_pred[0]))
    loss -= (positive_weights['mountains'] * y_true[1] * K.log(y_pred[1]) +
             negative_weights['mountains'] * (1 - y_true[1]) * K.log(1 - y_pred[1]))
    loss -= (positive_weights['sea'] * y_true[2] * K.log(y_pred[2]) +
             negative_weights['sea'] * (1 - y_true[2])* K.log(1 - y_pred[2]))
    loss -= (positive_weights['sunset'] * y_true[3] * K.log(y_pred[3]) +
             negative_weights['sunset'] * (1 - y_true[3]) * K.log(1 - y_pred[3]))
    loss -= (positive_weights['trees'] * y_true[4] * K.log(y_pred[4]) +
             negative_weights['trees'] * (1 - y_true[4]) * K.log(1 - y_pred[4]))

    return loss

class NatureDataset(Dataset):
    """multi-label dataset"""

    def __init__(self, train=True, augmentation=False, preprocessing_fn=None):
        self.train = train

        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn

        if self.train:
            self.all_files = train_df
        else:
            self.all_files = val_df

    def __len__(self):
        return self.all_files.shape[0]
    
    def __getitem__(self, idx):
    
        image = Image.open(self.all_files['filenames'][idx]) # use pillow to open a file
        label = self.all_files.iloc[idx][class_names].values.astype(np.float32)
        
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)
        
        return image, label

def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=5):

  for epoch in trange(num_epochs, desc="Epochs"):
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
        data, target = data.to(device), target.to(device)

        with torch.set_grad_enabled(phase=="train"):
          output = model(data)
          loss = criterion(output, target)
          preds = torch.sigmoid(output).data > 0.5
          preds = preds.to(torch.float32)
          
          if phase=="train"  :
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # statistics
        running_loss += loss.item() * data.size(0)
        running_corrects += f1_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="samples")  * data.size(0)
        
        
      epoch_loss = running_loss / len(data_loader[phase].dataset)
      epoch_acc = running_corrects / len(data_loader[phase].dataset)

      result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    print(result)


batch_size=32
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

dataset = NatureDataset(train=True, augmentation=True, preprocessing_fn=transform)
print('Dataset size:', len(dataset))
partition_point = int(len(dataset)*0.12) 
trainset, valset  = random_split(dataset, [len(dataset) - partition_point, partition_point])
data_loader = {"train": DataLoader(trainset , shuffle=True , batch_size=8),
                "val": DataLoader(valset, shuffle=True, batch_size=8)}

model = torchvision.models.resnet50(pretrained=True) # load the pretrained model
num_features = model.fc.in_features # get the no of on_features in last Linear unit
print("Number of features is the last Linear unit", num_features)
## freeze the entire convolution base
for param in model.parameters():
  param.requires_grad_(False)

def create_head(num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU):
  features_lst = [num_features , num_features//2 , num_features//4]
  layers = []
  for in_f, out_f in zip(features_lst[:-1] , features_lst[1:]):
    layers.append(nn.Linear(in_f , out_f))
    layers.append(activation_func())
    layers.append(nn.BatchNorm1d(out_f))
    if dropout_prob !=0 : layers.append(nn.Dropout(dropout_prob))
  layers.append(nn.Linear(features_lst[-1] , number_classes))
  return nn.Sequential(*layers)

NUM_CLASSES = 5
top_head = create_head(num_features , NUM_CLASSES) # because ten classes
model.fc = top_head # replace the fully connected layer
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = model.to(device)
print(model)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005 )

from tqdm import trange
from sklearn.metrics import precision_score, f1_score
train(model, data_loader, criterion, optimizer, sgdr_partial, num_epochs=10)
print("Done")