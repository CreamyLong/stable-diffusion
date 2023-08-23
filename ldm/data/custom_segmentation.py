from ldm.data.seg_dataset import SegmentationBase


class customTrain(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/sflckr_examples.txt",
                         data_root="data/sflckr_images",
                         segmentation_root="data/sflckr_segmentations",
                         size=size,
                         random_crop=random_crop,
                         interpolation=interpolation)

class customValidation(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv='data/flickr/flickr_eval.txt',
                         data_root='data/flickr/flickr_images',
                         segmentation_root='data/flickr/flickr_segmentations',
                         size=size,
                         random_crop=random_crop,
                         interpolation=interpolation)

class customTest(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv='data/flickr/flickr_eval.txt',
                         data_root='data/flickr/flickr_images',
                         segmentation_root='data/flickr/flickr_segmentations',
                         size=size,
                         random_crop=random_crop,
                         interpolation=interpolation)

class Examples(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/sflckr_examples.txt",
                         data_root="data/sflckr_images",
                         segmentation_root="data/sflckr_segmentations",
                         size=size,
                         random_crop=random_crop,
                         interpolation=interpolation)