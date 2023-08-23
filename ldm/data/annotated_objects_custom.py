import json
from itertools import chain
from pathlib import Path
from typing import Iterable, Dict, List, Callable, Any
from collections import defaultdict

from tqdm import tqdm

from ldm.data.annotated_objects_dataset import AnnotatedObjectsDataset
from ldm.data.conditional_builder.utils import load_annotations, load_categories
from ldm.data.helper_types import Annotation, ImageDescription, Category

from ldm.data.custom_config import custom_PATH_STRUCTURE


def load_image_descriptions(description_json: List[Dict]) -> Dict[str, ImageDescription]:
    return {
        str(img['id']): ImageDescription(
            id=img['id'],
            file_name=img['file_name'],
            original_size=(img['width'], img['height']),
        )
        for img in description_json
    }


class AnnotatedObjectCustom(AnnotatedObjectsDataset):
    def __init__(self, use_train: bool = True, use_val: bool = True, **kwargs):
        """
        @param data_path: is the path to the following folder structure:
                          apple/
                          ├── annotations
                          │   ├── instances_train2017.json
                          │   ├── instances_val2017.json
                          │   ├── stuff_train2017.json
                          │   └── stuff_val2017.json
                          ├── train2017
                          │   ├── 000000000009.jpg
                          │   ├── 000000000025.jpg
                          │   └── ...
                          ├── val2017
                          │   ├── 000000000139.jpg
                          │   ├── 000000000285.jpg
                          │   └── ...
        @param: split: one of 'train' or 'validation'
        @param: desired image size (give square images)
        """
        super().__init__(**kwargs)
        self.use_train = use_train
        self.use_val = use_val

        with open(self.paths['annotations']) as f:
            data_json = json.load(f)

        category_jsons = []
        annotation_jsons = []
        category_jsons.append(data_json['categories'])
        annotation_jsons.append(data_json['annotations'])

        self.categories = load_categories(chain(*category_jsons))
        self.filter_categories()
        self.setup_category_id_and_number()

        self.image_descriptions = load_image_descriptions(data_json['images'])

        annotations = load_annotations(annotation_jsons, self.image_descriptions, self.get_category_number, self.split)

        self.annotations = self.filter_object_number(annotations,
                                                     self.min_object_area,
                                                     self.min_objects_per_image,
                                                     self.max_objects_per_image)

        self.image_ids = list(self.annotations.keys())
        self.clean_up_annotations_and_image_descriptions()

    def get_path_structure(self) -> Dict[str, str]:
        if self.split not in custom_PATH_STRUCTURE:
            raise ValueError(f'Split [{self.split} does not exist for data.]')
        return custom_PATH_STRUCTURE[self.split]

    def get_image_path(self, image_id: str) -> Path:
        return self.paths['files'].joinpath(self.image_descriptions[str(image_id)].file_name)

    def get_image_description(self, image_id: str) -> Dict[str, Any]:
        # noinspection PyProtectedMember
        return self.image_descriptions[image_id]._asdict()
