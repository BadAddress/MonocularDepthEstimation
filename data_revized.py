from MLFocalLengths import estimate_focal_length
import time
import yaml
import os
from PIL import Image
from collections import defaultdict
import torch
import vanishing_point
import math
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from torchvision import transforms 
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
import torch.nn as nn
import debugTool
from coco_categories import COCO_CATEGORIES
import numpy as np
from tqdm import tqdm



class MyDataset(Dataset):
    def __init__(self,mode):
        # 654  test  
        # 24230 train
        self.mode = mode    
        if mode =="train":
            self.srcPath = "/home/bl/Desktop/bde/NPYDataSet"
        else:
            self.srcPath = "/home/bl/Desktop/bde/NPYDataSet_test"

        self.itemCount = len(os.listdir(self.srcPath))

        if self.mode == "train":
            self.bound = 24230 
        else:
            self.bound = 654

    def __getitem__(self, index):
        
        if index >= self.bound:
            raise IndexError("Index out of range")
        fileIDX = int(index) 
        filePath = os.path.join(self.srcPath,f"{fileIDX}.npy")
        data = np.load(filePath,allow_pickle=True)
        item = data[0]
        #debugTool.visualize_tensors(item[0],item[1],item[3],item[2])
        gt = item[3]/1000
        gt[gt>10] = 10.0
        return item[0],item[1],item[2],gt


    def __len__(self):
        return self.bound   



if __name__ == "__main__":

    test = MyDataset("train")

    test.__getitem__(11111)
    


    pass