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



sys.path.insert(0,"./Segment-Everything-Everywhere-All-At-Once")
from demo.seem import tools

with open("____config.yaml","r") as file:
    cfg = yaml.safe_load(file)

def create_map_fn(mapping):
    def map_fn(x):
        return mapping.get(int(x),0)
    return map_fn

class common_utils:
    def __init__(self) -> None:
        return None
    
    def dependent_initialization(self):
        self.evaluator = estimate_focal_length.Evaluator(cfg["MLFocalLengths_pth"])
        self.segInferencer = tools.SegementResGenerator(cfg["SEEM_config_path"],cfg["SEEM_pth"])
        print("\033[95m Common utilities and dependent inference model have been loaded successfully ! \033[0m")

    def generate_tensor_by_vanishing_point(self,image_path):
        #432 576
        height = int(cfg["img_height"])
        width = int(cfg["img_width"])
        # _,focal_length = self.evaluator.evaluate(image_path)
        focal_length = 0
        vp = vanishing_point.detect_and_mark_vanishing_point(image_path,focal_length,cfg["VanishingPoint_Viz"],1,False)
        a_indices = torch.arange(0, height).view(-1, 1).repeat(1, width)
        b_indices = torch.arange(0, width).repeat(height, 1)
        delta_y = a_indices - vp[1]
        delta_x = b_indices - vp[0]
        
        distance = torch.sqrt(delta_x**2 + delta_y**2)
        max_dist_threshold = 1.5 * (width*2 + height**2)**0.5
        max_dist_threshold = torch.tensor(max_dist_threshold)
        clamped_distance = torch.clamp(distance, max=max_dist_threshold)
        distance = clamped_distance / max_dist_threshold

        # 计算角度
        # angle = torch.atan2(delta_y, delta_x) % (2 * math.pi)
        angle = torch.atan2(delta_y, delta_x)
        angle = torch.sin(angle)
        angle = torch.pow(angle,2)

        return distance, angle


class MyDataset(Dataset):
    def __init__(self,mode):
        self.utl = common_utils()
        self.utl.dependent_initialization()
        self.basePath = cfg["DataSetBasePath"]
        self.mode = mode    
        if mode =="train":
            self.srcGuide = "nyu_train.txt"
        else:
            self.srcGuide = "nyu_test.txt"
        self.srcFile = os.path.join(self.basePath,self.srcGuide)
        
        with open(self.srcFile, 'r') as f:
            self.srcList = f.readlines()
        self.srcList = sorted(self.srcList)

        self.preTransforms = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, index):
        if index >= len(self.srcList):
            raise IndexError("Index out of range")
        try:
            splitedItem = self.srcList[index].split()
            if self.mode == "test":
                inputRGB = self.basePath +"/"+ splitedItem[0]
                gt = self.basePath + "/"+splitedItem[1]
            else:
                inputRGB = self.basePath + splitedItem[0]
                gt = self.basePath + splitedItem[1]
            
            
            dis_tensor, theta_tensor = self.utl.generate_tensor_by_vanishing_point(inputRGB)
            pano_seg_tensor, pano_seg_info= self.utl.segInferencer.getSegRes(inputRGB)
            
            pano_seg_pil = to_pil_image(pano_seg_tensor)
            pano_seg_resized_pil = resize(pano_seg_pil, size=[432, 576])
            pano_seg_tensor = to_tensor(pano_seg_resized_pil)
            pano_seg_tensor = pano_seg_tensor.squeeze(0)

            idmp = {}
            for item in pano_seg_info:
                k = item['id']
                v = item['category_id']
                if v>200 :
                    v = 0
                idmp[k]=v

            custom_map_fn = create_map_fn(idmp)

            pano_seg_tensor.to(torch.float64)
            pano_seg_tensor.apply_(custom_map_fn)
            pano_seg_tensor.to(torch.int32)
            dis_tensor  = dis_tensor.unsqueeze(0)
            theta_tensor = theta_tensor.unsqueeze(0)
            pano_seg_tensor = pano_seg_tensor.unsqueeze(0)
            
            transform = transforms.ToTensor()
            image = Image.open(inputRGB)
            image_tensor = transform(image)        
            gt = Image.open(gt)
            gt_tensor = transform(gt)
            vp_tensor = torch.cat((dis_tensor,theta_tensor),dim=0)
            return vp_tensor,pano_seg_tensor,image_tensor,gt_tensor
        except :
            raise ValueError("Preprocess Error")
        
            pass

        debugTool.visualize_tensors(dis_tensor,theta_tensor,pano_seg_tensor,gt_tensor,image_tensor)
        stack_tensors = torch.stack([dis_tensor,theta_tensor,pano_seg_tensor],dim=0)
        final_tensor = torch.cat([image_tensor,stack_tensors],dim=0)
        print(final_tensor.shape)
        

    def __len__(self):
        return len(self.srcList)    



if __name__ == "__main__":

    ds = MyDataset("train")
    i = 0
    data_buffer = []
    save_dir = "/home/bl/Desktop/bde/NPYDataSet"

    print( len(ds.srcList) )

    for item in tqdm(ds,desc="Processing items"):
        try:
            data_buffer.append(item)
            if len(data_buffer) == 1:
                data_buffer_np = np.empty(len(data_buffer), dtype=object)
                for idx, val in enumerate(data_buffer):
                    data_buffer_np[idx] = val
                np.save(f'{save_dir}/{i}.npy', data_buffer_np)
                data_buffer = []
                i += 1
        except IndexError:
            print("FINISH ITERATION")
            break
        except ValueError:
            print("SKIPPED BAD SAMPLE")
            continue



    if data_buffer:
        data_buffer_np = np.empty(len(data_buffer), dtype=object)
        for idx, val in enumerate(data_buffer):
            data_buffer_np[idx] = val
        np.save(f'{save_dir}/{i}.npy', data_buffer_np)