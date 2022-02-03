import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class MiniPlaces(Dataset):
    def __init__(self, split='train', root_dir=''):
        super(MiniPlaces, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = root_dir 
        self.data = []

        """
        Implement the data initialization process.
        self.root_dir is the assignment directory.
        self.data is a list of (image_ids, label, label_text) tuples which contains all images and labels of that split.
        You can find the information in train.txt and val.txt.
        image_ids help you retrieve the images in __getitem__, it can be absolute / relative path to the image.
        label is integar label (hint: use line.strip().split() to find ids and labels)
        label_text is the name of the category (e.g., abbey).

        For validation set, label_text can be just set to 0 because we do not have this information.

        For test set, you do not need labels and label texts.
        For test set, you have to use os.listdir(...) to find all image ids.

        """
        #####################################################Your Code######################################
        #pass
        ####################################################################################################

        """
        Define self.img_transform = ...
        First transform data to tensor and then normalize. mean and variance: (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        """
        #####################################################Your Code######################################
        #pass
        ####################################################################################################

    def __getitem__(self, index):
        """
        Given an index, return the image, label and label text at that index.
        For image, you should use PIL to open it and convert to RGB.
        RESIZE it to be 64*64.
        Then use self.img_transform to transform it.
        permute each image so that each image has size (64, 64, 3).
        """
        image, label, label_text = None, None, None
        #####################################################Your Code######################################
        #pass
        ####################################################################################################
        return image, label, label_text
    
    def __len__(self):
        return len(self.data)