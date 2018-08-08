import os
import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset

from . import data_transforms


class BDDDataset(Dataset):
    def __init__(self, list_path, img_size, is_training, is_debug=False):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt'
                            ).replace('.jpg', '.txt') for path in
                            self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.is_debug = is_debug

        #  transforms and augmentation
        self.transforms = data_transforms.Compose()
        if is_training:
            self.transforms.add(data_transforms.ImageBaseAug())
        # self.transforms.add(data_transforms.KeepAspect())
        self.transforms.add(data_transforms.ResizeImage(self.img_size))
        self.transforms.add(data_transforms.ToTensor(self.max_objects, self.is_debug))

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
        else:
            logging.info("label does not exist: {}".format(label_path))
            labels = np.zeros((1, 5), np.float32)

        sample = {'image': img, 'label': labels}
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample

    def __len__(self):
        return len(self.img_files)
