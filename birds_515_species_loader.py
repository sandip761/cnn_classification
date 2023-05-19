from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

class BirdsSpeciesDataset(Dataset):
    

    def __init__(self, data_frame, root_dir, transform=None):

        self.data_frame = data_frame;
        self.root_dir = root_dir;
        self.transform = transform;

    def __len__(self):
        return len(self.data_frame)
    

    def __getitem__(self, idx) :
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        return super().__getitem__(idx)
    
birds_df = pd.read_csv('../birds.csv')
birds_train_df = birds_df[birds_df['data set']=="train"]
#print(len(birds_train_df))
#print(len(birds_df))
dataset_train = BirdsSpeciesDataset(birds_df, root_dir="../train", transform=None)
print(dataset_train.__len__())