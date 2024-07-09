# -*- coding: utf-8 -*-

import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import time
import cv2
import json
import albumentations


# -*- coding: utf-8 -*-

import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import time
import cv2
import json
import albumentations

class CustomBase(Dataset):
    def __init__(self, img_paths, txt_paths,  size=None, random_crop=False, random_flip=False, random_rotate=False):

        self.size = size
        self.img_paths = img_paths
        self.txt_paths = txt_paths
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate

        assert len(self.img_paths) == len(self.txt_paths)

        self._length = len(img_paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            if self.random_flip:
                self.flipor = albumentations.HorizontalFlip(p=0.5)
            if self.random_rotate:
                self.rotator = albumentations.RandomRotate90(p=0.5)
            
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper, self.flipor, self.rotator])
        else:
            self.preprocessor = lambda **kwargs: kwargs


    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


    def __len__(self):
        return len(self.img_paths) #我修改的

    def __getitem__(self, i):
        example = {}

        example["image"] = self.preprocess_image(self.img_paths[i])
        example["caption"] = self.txt_paths[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file="data/coco_images.txt", training_txt_list_file="data/coco_txt.txt"):
       
        self.size = size

        with open(training_images_list_file, "r") as f:
            self.img_paths = f.read().splitlines()

        with open(training_txt_list_file, "r") as f:
            self.txt_paths = f.read().splitlines()

        super().__init__(img_paths=self.img_paths, txt_paths=self.txt_paths,size=self.size)

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file="data/coco_images.txt", test_classes_list_file="data/coco_txt.txt"):
        
        self.size = size

        with open(test_images_list_file, "r") as f:
            self.img_paths = f.read().splitlines()

        with open(test_classes_list_file, "r") as f:
            self.txt_paths = f.read().splitlines()

        super().__init__(img_paths=self.img_paths, txt_paths=self.txt_paths,size=self.size)



