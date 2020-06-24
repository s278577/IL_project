from PIL import Image
import os
import os.path
import numpy as np
import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets import CIFAR100
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

class Cifar100(CIFAR100):
    def __init__(self, root = 'Dataset', classes=range(10), train=True, transform=None, target_transform=None, download=True):
        
        super(Cifar100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # Select subset of classes
        
        data = []
        targets = []

        for i in range(len(self.data)):
            if self.targets[i] in classes:
                data.append(self.data[i])
                targets.append(self.targets[i])

        self.data = np.array(data)
        self.targets = targets


    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
       
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        
        return len(self.data)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

    def append(self, images, labels):

        self.data = np.concatenate((self.data, images), axis=0)
        self.targets = self.targets + labels