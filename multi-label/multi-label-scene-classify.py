""" Multi-label images classification, i.e., one images can contain multiple labels"""
import os
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torchvision import models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from scipy.io import loadmat  # Load MATLAB file.
from sklearn.metrics import f1_score, roc_auc_score

print(f'PyTorch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')


class SceneDataset(Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()
        self.df = df
        self.transforms = transforms

    def get_labels(self, idx):
        record = self.df.iloc[idx]
        return record[1:].tolist()

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        image = Image.open(record['filename']).convert("RGB")
        label = torch.tensor(record[1:].tolist(), dtype=torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.df)


class ExtendedResNetModel(nn.Module):
    """ Extend ResNet with three new fully connected layers and attach them as a head to a ResNet50 trunk"""

    def __init__(self, nb_classes, dropout_prob=0.3, activation_func=nn.ReLU):
        super().__init__()
        # load the pretrained model as feafures
        self.rn50_features = models.resnet50(pretrained=True)
        # get the nb of in_features in last Linear unit
        nb_in_features_last = self.rn50_features.fc.in_features
        for param in self.rn50_features.parameters():
            param.requires_grad_(False)

        head = self._create_head(nb_in_features_last, nb_classes,
                                 dropout_prob, activation_func)
        self.rn50_features.fc = head  # attach head
        # print(self.rn50_features)

    def _create_head(self, nb_features, nb_classes, dropout_prob=0.3, activation_func=nn.ReLU):
        features_lst = [nb_features, nb_features//2, nb_features//4]
        layers = []
        for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.BatchNorm1d(out_f))
            layers.append(activation_func())
            if dropout_prob != 0:
                layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1], nb_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.rn50_features(x)
        return x


def train(model, data_loader, criterion, optimizer, scheduler, nb_epochs=5):

    for epoch in range(nb_epochs):
        result = []
        for phase in ['train', 'val']:
            if phase == "train":  
                model.train()  # put model in training mode
            else:  
                model.eval()  # # put model in validation mode

            # Track for each epoch
            running_loss = 0.0
            running_f1_score = 0.0
            running_roc_auc_score = 0.0

            for data, targets in data_loader[phase]:
                data, targets = data.to(device), targets.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(data)  # forward pass
                    loss = criterion(outputs, targets)
                    # preds = torch.sigmoid(output).data > 0.5
                    preds = outputs.data > 0.5
                    preds = preds.to(torch.float32)

                    if phase == "train":
                        loss.backward()  # compute gradient of the loss with respect to model parameters
                        optimizer.step()  # update the model parameters
                        scheduler.step()
                        optimizer.zero_grad()

                running_loss += loss.item() * len(data)
                running_f1_score += f1_score(targets.to("cpu").to(torch.int).numpy(),
                                             preds.to("cpu").to(torch.int).numpy(),
                                             average="samples") * len(data)
                running_roc_auc_score += roc_auc_score(targets.to("cpu").to(torch.int).numpy(),
                                                       preds.to("cpu").to(torch.int).numpy(),
                                                       average="samples") * len(data)

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_f1_score = running_f1_score / len(data_loader[phase].dataset)
            epoch_roc_auc_score = running_roc_auc_score / \
                len(data_loader[phase].dataset)

            result.append(f'Epoch:{epoch} {phase.upper()}: Loss:{epoch_loss:.4f} '
                          f'F1-Score: {epoch_f1_score:.4f} AUC: {epoch_roc_auc_score:.4f}')
        print(result)

# To download the dataset, see accompanying file, dataset_download_steps.txt.
dataset_root = '.'

# Read the "processed" part for class names and class labels
processed_mat = loadmat(os.path.join(dataset_root, 'miml data.mat'))
class_labels = []
for c in processed_mat['class_name']:  # get the name of each class
    class_labels.append(c[0][0])
nb_classes = len(class_labels)
print('class labels:', class_labels)  # ['desert', 'mountains', 'sea', 'sunset', 'trees']

# Read the labels. If multi-class label for ith images equals [1, -1, -1, 1, -1], it means:
# i-th image belongs to the 1st & 4th class but does not belong to the 2nd, 3rd &  5th classes
labels = copy.deepcopy(processed_mat['targets'].T)
labels[labels == -1] = 0  # convert to range [0, 1] from [-1, 1]

# Setup a pandas dataframe with file location and associated (multi) labels as below
#                   filename desert mountains sea sunset trees
# 0  ./original_images/1.jpg      1         0   0      0     0
# 1  ./original_images/2.jpg      1         0   0      0     0
# 2  ./original_images/3.jpg      1         0   0      0     0
# 3  ./original_images/4.jpg      1         1   0      0     0
# 4  ./original_images/5.jpg      1         0   0      0     0

# create empty dataframe with columns, [filename desert mountains sea sunset tree]
data_df = pd.DataFrame(columns=['filename'] + class_labels)
filenames = os.listdir(os.path.join(dataset_root, "original_images/"))
data_df['filename'] = np.array(
    sorted(list(map(lambda x: int(Path(x).stem), np.array(filenames)))))
data_df['filename'] = data_df['filename'].apply(
    lambda x: os.path.join(dataset_root, 'original_images/') + str(x) + '.jpg')
data_df[class_labels] = np.array(labels)

transforms_list = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])])

split_ratio = 0.3
dataset = SceneDataset(data_df, transforms_list)
split_point = int(len(dataset) * split_ratio)
trainset, valset = random_split(
    dataset, [len(dataset) - split_point, split_point], generator=torch.Generator().manual_seed(0))
print(f"Train set size: {len(trainset)}; Val set size: {len(valset)}")

batch_size = 128
dataloader = {"train": DataLoader(trainset, shuffle=True, batch_size=batch_size),
              "val": DataLoader(valset, shuffle=True, batch_size=batch_size)}

positive_weights = []
for cls in range(nb_classes):
    positive_samples = float(sum([dataset.get_labels(idx)[cls] == 1 for idx in trainset.indices]))
    negative_samples = float(sum([dataset.get_labels(idx)[cls] == 0 for idx in trainset.indices]))
    pos_weight = negative_samples / positive_samples
    positive_weights.append(pos_weight)
positive_weights = torch.FloatTensor(positive_weights).to('cuda')
print('Ratio of Negative samples to positive samples per class:', positive_weights)

torch.manual_seed(0)
model = ExtendedResNetModel(nb_classes=nb_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
sgdr_cos_anneal_sched = lr_scheduler.CosineAnnealingLR(optimizer,  # set learning rate schedule
                                                 T_max=5, eta_min=0.005)

train(model, dataloader, criterion, optimizer, sgdr_cos_anneal_sched, nb_epochs=10)
