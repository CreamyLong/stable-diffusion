from ldm.data.annotated_objects_custom import AnnotatedObjectCustom


class customTrain(AnnotatedObjectCustom):
    def __init__(self, size):
        super().__init__(data_path='./data/',
                         split='train',
                         keys=['image', 'objects_bbox'],
                         no_tokens=8192,
                         target_image_size=size,
                         min_object_area=0.00001,
                         min_objects_per_image=1,
                         max_objects_per_image=30,
                         crop_method='center',
                         random_flip=False,
                         use_group_parameter=False,
                         encode_crop=False)


class customValidation(AnnotatedObjectCustom):
    def __init__(self, size):
        super().__init__(data_path='./data/',
                         split='validation',
                         keys=['image', 'objects_bbox'],
                         no_tokens=8192,
                         target_image_size=size,
                         min_object_area=0.00001,
                         min_objects_per_image=1,
                         max_objects_per_image=30,
                         crop_method='center',
                         random_flip=False,
                         use_group_parameter=False,
                         encode_crop=False)


class customTest(AnnotatedObjectCustom):
    def __init__(self, size):
        super().__init__(data_path='./data/',
                         split='test',
                         keys=['image', 'objects_bbox'],
                         no_tokens=8192,
                         target_image_size=size,
                         min_object_area=0.00001,
                         min_objects_per_image=1,
                         max_objects_per_image=30,
                         crop_method='center',
                         random_flip=False,
                         use_group_parameter=False,
                         encode_crop=False)