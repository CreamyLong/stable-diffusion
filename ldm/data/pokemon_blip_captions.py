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
from pandas import read_parquet

import io
from PIL import Image


def byte2image(byte_data):
    image = Image.open(io.BytesIO(byte_data))
    return image


class CustomBase(Dataset):
    def __init__(self,  size, data=None, random_crop=False, random_flip=False, random_rotate=False):


        self.size = size
        self.data = read_parquet(data)
        self.img_paths =  self.data['image']
        self.txt_paths =  self.data['text']

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate

        assert len(self.img_paths) == len(self.txt_paths)

        self._length = len(self.img_paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler])

        #     if not self.random_crop:
        #         self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        #     else:
        #         self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
        #     if self.random_flip:
        #         self.flipor = albumentations.HorizontalFlip(p=0.5)
        #     if self.random_rotate:
        #         self.rotator = albumentations.RandomRotate90(p=0.5)
        #     self.preprocessor = albumentations.Compose([self.rescaler, self.cropper, self.flipor, self.rotator])
        # else:
        #     self.preprocessor = lambda **kwargs: kwargs

    def preprocess_image(self, image_path):

        image = byte2image(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = {}
        example["image"] = self.preprocess_image(self.img_paths[i]['bytes'])
        example["caption"] = self.txt_paths[i]
        return example


class CustomTrain(CustomBase):
    def __init__(self, size, data=None, random_crop=False, random_flip=False, random_rotate=False):
        super().__init__(
            size=size,
            data=data,
            random_crop=random_crop,
            random_flip=random_flip,
            random_rotate=random_rotate
        )


class CustomTest(CustomBase):
    def __init__(self, size, data=None):
        super().__init__(
            size=size,
            data=data,
            random_crop=False,
            random_flip=False,
            random_rotate=False
        )




