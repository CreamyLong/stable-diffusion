import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from ldm.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file="data/train.txt", training_classes_list_file=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()

        with open(training_classes_list_file, "r") as f:
            self.class_labels = f.read().splitlines()
        labels = {
            "class_label": self.class_labels,
        }
        self.data = ImagePaths(paths=paths,
                               labels=labels,
                               size=size,
                               random_crop=False)

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file="data/test.txt", test_classes_list_file=None):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()

        with open(test_classes_list_file, "r") as f:
            self.class_labels = f.read().splitlines()
        labels = {
            "class_label": self.class_labels,
        }
        self.data = ImagePaths(paths=paths,
                               labels=labels,
                               size=size,
                               random_crop=False)

