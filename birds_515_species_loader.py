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