# -*- coding: utf-8 -*-

import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from ldm.data.base import ImagePaths


class CustomBase(Dataset):
    def __init__(self, size, txt_file="data/train.txt"):
        super().__init__()
        self.class_labels = []
        self.txt_file = txt_file

        with open(txt_file, "r") as f:
            self.paths = f.read().splitlines()

        for p in self.paths:
            name = p.split('/')[-2].split('.')[-1]
            self.class_labels.append(name)
        sorted_classes = {x: i for i, x in enumerate(sorted(set(self.class_labels)))}
        classes = [sorted_classes[x] for x in self.class_labels]
        print(sorted_classes)
        labels = {
            "class_label": np.array(classes),
        }

        self.data = ImagePaths(paths=self.paths,
                               labels=labels,
                               size=size,
                               random_crop=False,
                               random_flip=False,
                               random_rotate=False)
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file="data/train.txt"):
        super().__init__(size, training_images_list_file)
        self.size = size
        self.txt_file = training_images_list_file


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file="data/test.txt"):
        super().__init__(size, test_images_list_file)

        self.size = size
        self.txt_file = test_images_list_file
