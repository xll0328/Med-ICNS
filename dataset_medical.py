from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
import os
import numpy as np
import torch
import random
class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None,shuffle=False):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        self.targets = [] 
        self.shuffle=shuffle
        for c in range(self.num_cls):
            
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            # if shuffle:
            #     cls_list=random.sample(cls_list,len(cls_list))
            self.img_list += cls_list
            self.targets += [int(item[1]) for item in cls_list]  # 将标签添加到 targets
        if shuffle:
            paired_arrays = list(zip(self.img_list, self.targets))
            random.shuffle(paired_arrays)
            self.img_list,self.targets=zip(*paired_arrays)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = int(self.img_list[idx][1])
        # sample = {'img': image,
        #           'label': int(self.img_list[idx][1])}
        return image,label



def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


