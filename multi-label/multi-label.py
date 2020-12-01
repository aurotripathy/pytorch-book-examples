# https://medium.com/the-owl/imbalanced-multilabel-image-classification-using-keras-fbd8c60d7a4b
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil, os, time, random, copy
import imageio
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, confusion_matrix, average_precision_score
from skimage.transform import rotate, AffineTransform, warp, resize
from skimage import io
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pudb import set_trace

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

data_df = pd.DataFrame(columns=["filenames"] + class_names)
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
img = imageio.imread(os.path.join(content_root, 'original_images/1.jpg'))
H, W, _ = img.shape
print(f'image shape: {H}, {W}')

class NatureDataset(Dataset):
    """multi-label dataset"""

    def __init__(self, train=True, augmentation=False, preprocessing_fn=None, batch_size= 16):
        self.train = train
        self.batch_size = batch_size
        self.H = H
        self.W = W

        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn

        if self.train:
            self.all_files = train_df
        else:
            self.all_files = val_df

    def __len__(self):
        return self.all_files.shape[0] // self.batch_size
    
    def on_epoch_end(self):
        self.all_files = self.all_files.sample(frac=1).reset_index(drop=True)
    
    def __getitem__(self, idx):
        images = np.array([], dtype=np.float32).reshape((0, self.H, self.W, 3))
        labels = np.array([], dtype=np.float32).reshape((0, 5))
        for i in range(self.batch_size):
            # image = img_to_array(load_img(self.all_files['filenames'][idx * self.batch_size+i],
            #                               target_size=(self.H, self.W)))
            # https://stackoverflow.com/questions/50420168/how-do-i-load-up-an-image-and-convert-it-to-a-proper-tensor-for-pytorch
            # image = Image.open(self.all_files['filenames'][idx * self.batch_size + i]) # use pillow to open a file
            image = io.imread(self.all_files['filenames'][idx * self.batch_size + i]) # use pillow to open a file
            image = resize(image, (self.H, self.W))
            set_trace()

            y = self.all_files.iloc[idx * self.batch_size+i][class_names].values.astype(np.float32)
      
            # If there is any transform method, apply it onto the image
            if self.augmentation:
                image = rotate(image, np.random.uniform(-30, 30), preserve_range=True)

                # image = image.rotate(np.random.uniform(-30, 30), expand=False)
                scale = np.random.uniform(1.0, 1.25)
                tx = np.random.uniform(0, 20)
                ty = np.random.uniform(0, 20)
                image = warp(image,
                             AffineTransform(matrix=np.array([[scale, 0, tx],
                                                              [0,scale,  ty],
                                                              [0,   0,   1]])).inverse,
                             preserve_range=True)
            #RANDOM HORIZONTAL FLIPPING
            if np.random.choice([True, False]):
                image = np.flip(image, axis= 1)
            images = np.append(images, np.expand_dims(image, axis=0), axis=0)
            labels = np.append(labels,y.reshape(1, 5), axis=0)
        
        if self.preprocessing_fn:
            images = self.preprocessing_fn(images)
        
        return images, labels

train_dataset = NatureDataset(train=True, augmentation=True, preprocessing_fn=None, batch_size=8)

for i in range(len(train_dataset)):
    images, labels = train_dataset[i]
    print(len(images), len(labels))
