import os
from abc import abstractmethod
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
import bisect
import numpy as np
import albumentations
from PIL import Image

import cv2


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
        Define an interface to make the IterableDatasets for text2img data chainable
    '''

    def __init__(self, file_path: str, rank, world_size):
        super().__init__()
        self.file_path = file_path
        self.folder_list = []
        self.file_list = []
        self.txt_list = []
        self.info = self._get_file_info(file_path)
        self.start = self.info['start']
        self.end = self.info['end']
        self.rank = rank

        self.world_size = world_size
        # self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        # self.iter_start = self.start + self.rank * self.per_worker
        # self.iter_end = min(self.iter_start + self.per_worker, self.end)
        # self.num_records = self.iter_end - self.iter_start
        # self.valid_ids = [i for i in range(self.iter_end)]
        self.num_records = self.end - self.start
        self.valid_ids = [i for i in range(self.end)]

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        # return self.iter_end - self.iter_start
        return self.end - self.start

    def __iter__(self):
        sample_iterator = self._sample_generator(self.start, self.end)
        # sample_iterator = self._sample_generator(self.iter_start, self.iter_end)
        return sample_iterator

    def _sample_generator(self, start, end):
        for idx in range(start, end):
            file_name = self.file_list[idx]
            txt_name = self.txt_list[idx]
            f_ = open(txt_name, 'r')
            txt_ = f_.read()
            f_.close()
            image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image) / 255
            yield {"txt": txt_, "image": image}

    def _get_file_info(self, file_path):
        info = \
        {
            "start": 1,
            "end": 0,
        }
        self.folder_list = [file_path + i for i in os.listdir(file_path) if '.' not in i]
        for folder in self.folder_list:
            files = [folder + '/' + i for i in os.listdir(folder) if 'jpg' in i]
            txts = [k.replace('jpg', 'txt') for k in files]

            self.file_list.extend(files)
            self.txt_list.extend(txts)
        info['end'] = len(self.file_list)
        # with open(file_path, 'r') as fin:
        #     for _ in enumerate(fin):
        #         info['end'] += 1
        # self.txt_list = [k.replace('jpg', 'txt') for k in self.file_list]
        return info



class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, random_flip=False, random_rotate=False, labels=None):
        self.size = size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_rotate = random_rotate

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

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

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
