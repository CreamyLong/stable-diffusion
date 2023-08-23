import importlib
from typing import List, Any, Tuple, Optional

from ldm.data.helper_types import BoundingBox, Annotation

# source: seaborn, color palette tab10
COLOR_PALETTE = [(30, 118, 179), (255, 126, 13), (43, 159, 43), (213, 38, 39), (147, 102, 188),
                 (139, 85, 74), (226, 118, 193), (126, 126, 126), (187, 188, 33), (22, 189, 206)]
BLACK = (0, 0, 0)
GRAY_75 = (63, 63, 63)
GRAY_50 = (127, 127, 127)
GRAY_25 = (191, 191, 191)
WHITE = (255, 255, 255)
FULL_CROP = (0., 0., 1., 1.)


def intersection_area(rectangle1: BoundingBox, rectangle2: BoundingBox) -> float:
    """
    Give intersection area of two rectangles.
    @param rectangle1: (x0, y0, w, h) of first rectangle
    @param rectangle2: (x0, y0, w, h) of second rectangle
    """
    rectangle1 = rectangle1[0], rectangle1[1], rectangle1[0] + rectangle1[2], rectangle1[1] + rectangle1[3]
    rectangle2 = rectangle2[0], rectangle2[1], rectangle2[0] + rectangle2[2], rectangle2[1] + rectangle2[3]
    x_overlap = max(0., min(rectangle1[2], rectangle2[2]) - max(rectangle1[0], rectangle2[0]))
    y_overlap = max(0., min(rectangle1[3], rectangle2[3]) - max(rectangle1[1], rectangle2[1]))
    return x_overlap * y_overlap


def horizontally_flip_bbox(bbox: BoundingBox) -> BoundingBox:
    return 1 - (bbox[0] + bbox[2]), bbox[1], bbox[2], bbox[3]


def absolute_bbox(relative_bbox: BoundingBox, width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Convert relative bounding box to absolute bounding box.
    """
    bbox = relative_bbox
    bbox = bbox[0] * width, bbox[1] * height, (bbox[0] + bbox[2]) * width, (bbox[1] + bbox[3]) * height
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def pad_list(list_: List, pad_element: Any, pad_to_length: int) -> List:
    return list_ + [pad_element for _ in range(pad_to_length - len(list_))]


def rescale_annotations(annotations: List[Annotation], crop_coordinates: BoundingBox, flip: bool) -> List[Annotation]:
    def clamp(x: float):
        return max(min(x, 1.), 0.)  # clamp to [0, 1]

    def rescale_bbox(bbox: BoundingBox) -> BoundingBox:
        x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
        y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
        w = min(bbox[2] / crop_coordinates[2], 1 - x0)
        h = min(bbox[3] / crop_coordinates[3], 1 - y0)
        if flip:
            x0 = 1 - (x0 + w)
        return x0, y0, w, h

    return [a._replace(bbox=rescale_bbox(a.bbox)) for a in annotations]


def filter_annotations(annotations: List[Annotation], crop_coordinates: BoundingBox) -> List:
    return [a for a in annotations if intersection_area(a.bbox, crop_coordinates) > 0.0]


def additional_parameters_string(annotation: Annotation, short: bool = True) -> str:
    sl = slice(1) if short else slice(None)
    string = ''
    if not (annotation.is_group_of or annotation.is_occluded or annotation.is_depiction or annotation.is_inside):
        return string
    if annotation.is_group_of:
        string += 'group'[sl] + ','
    if annotation.is_occluded:
        string += 'occluded'[sl] + ','
    if annotation.is_depiction:
        string += 'depiction'[sl] + ','
    if annotation.is_inside:
        string += 'inside'[sl]
    return '(' + string.strip(",") + ')'


def get_plot_font_size(font_size: Optional[int], figure_size: Tuple[int, int]) -> int:
    if font_size is None:
        font_size = 10
        if max(figure_size) >= 256:
            font_size = 12
        if max(figure_size) >= 512:
            font_size = 15
    return font_size


def get_circle_size(figure_size: Tuple[int, int]) -> int:
    circle_size = 2
    if max(figure_size) >= 256:
        circle_size = 3
    if max(figure_size) >= 512:
        circle_size = 4
    return circle_size


def load_object_from_string(object_string: str) -> Any:
    """
    Source: https://stackoverflow.com/a/10773699
    """
    module_name, class_name = object_string.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)

from typing import Iterable, Dict, List, Callable, Any
from collections import defaultdict
from ldm.data.helper_types import Annotation, ImageDescription, Category
from itertools import chain
from tqdm import tqdm

def load_categories(category_json: Iterable) -> Dict[str, Category]:
    return {str(cat['id']): Category(id=str(cat['id']),
                                     super_category=cat['supercategory'],
                                     name=cat['name'])
            for cat in category_json if cat['name'] != 'other'}


def load_annotations(annotations_json: List[Dict],
                     image_descriptions: Dict[str, ImageDescription],
                     category_no_for_id: Callable[[str], int],
                     split: str) -> Dict[str, List[Annotation]]:

    annotations = defaultdict(list)
    total = sum(len(a) for a in annotations_json)

    for ann in tqdm(chain(*annotations_json), f'Loading {split} annotations', total=total):
        image_id = str(ann['image_id'])
        if image_id not in image_descriptions:
            raise ValueError(f'image_id [{image_id}] has no image description.')
        category_id = ann['category_id']
        try:
            category_no = category_no_for_id(str(category_id))
        except KeyError:
            continue

        width, height = image_descriptions[image_id].original_size

        # print("原有数据集中做的bbox", ann['bbox'], width, height)

        bbox = (ann['bbox'][0] / width, ann['bbox'][1] / height, ann['bbox'][2] / width, ann['bbox'][3] / height)
        # print("转换后的数据集中做的bbox", bbox)

        annotations[image_id].append(
            Annotation(
                id=ann['id'],
                area=bbox[2] * bbox[3],  # use bbox area
                is_group_of=ann['iscrowd'],
                image_id=ann['image_id'],
                bbox=bbox,
                category_id=str(category_id),
                category_no=category_no
            )
        )
    return dict(annotations)

