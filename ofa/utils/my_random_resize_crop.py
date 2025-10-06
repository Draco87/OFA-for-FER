import random
import math
from PIL import Image

import torchvision.transforms.functional as F
import torchvision.transforms as transforms

__all__ = ["MyRandomResizedCrop", "MyResizeRandomCrop", "MyResize"]

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}

class MyRandomResizedCrop(transforms.RandomResizedCrop):
    ACTIVE_SIZE = 128
    IMAGE_SIZE_LIST = [64, 80, 96, 128]
    IMAGE_SIZE_SEG = 4

    CONTINUOUS = False
    SYNC_DISTRIBUTED = False

    EPOCH = 0
    BATCH = 0

    def __init__(
        self,
        size,
        scale=(0.5, 1.0),
        ratio=(0.9, 1.1),
        interpolation=Image.BICUBIC,
    ):
        if not isinstance(size, int):
            size = size[0]
        super(MyRandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(
            img,
            i,
            j,
            h,
            w,
            (MyRandomResizedCrop.ACTIVE_SIZE, MyRandomResizedCrop.ACTIVE_SIZE),
            self.interpolation,
        )

    @staticmethod
    def get_candidate_image_size():
        if MyRandomResizedCrop.CONTINUOUS:
            min_size = min(MyRandomResizedCrop.IMAGE_SIZE_LIST)
            max_size = max(MyRandomResizedCrop.IMAGE_SIZE_LIST)
            candidate_sizes = []
            for i in range(min_size, max_size + 1):
                if i % MyRandomResizedCrop.IMAGE_SIZE_SEG == 0:
                    candidate_sizes.append(i)
        else:
            candidate_sizes = MyRandomResizedCrop.IMAGE_SIZE_LIST

        relative_probs = None
        return candidate_sizes, relative_probs

    @staticmethod
    def sample_image_size(batch_id=None):
        if batch_id is None:
            batch_id = MyRandomResizedCrop.BATCH
        random.seed(batch_id + MyRandomResizedCrop.EPOCH)
        candidate_sizes, relative_probs = MyRandomResizedCrop.get_candidate_image_size()
        MyRandomResizedCrop.ACTIVE_SIZE = random.choices(
            candidate_sizes, weights=relative_probs
        )[0]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(
            MyRandomResizedCrop.IMAGE_SIZE_LIST
        )
        if MyRandomResizedCrop.CONTINUOUS:
            format_string += "@continuous"
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string

class MyResizeRandomCrop(object):
    def __init__(
        self,
        interpolation=Image.BICUBIC,
        use_padding=False,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
    ):
        self.interpolation = interpolation
        self.use_padding = use_padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        crop_size = MyRandomResizedCrop.ACTIVE_SIZE
        img = F.resize(img, crop_size, self.interpolation)
        i, j, h, w = transforms.RandomCrop.get_params(img, (crop_size, crop_size))
        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return (
            "MyResizeRandomCrop(size=%s%s, interpolation=%s, use_padding=%s, fill=%s)"
            % (
                MyRandomResizedCrop.IMAGE_SIZE_LIST,
                "@continuous" if MyRandomResizedCrop.CONTINUOUS else "",
                _pil_interpolation_to_str[self.interpolation],
                self.use_padding,
                self.fill,
            )
        )

class MyResize(object):
    def __init__(self, interpolation=Image.BICUBIC):
        self.interpolation = interpolation

    def __call__(self, img):
        target_size = MyRandomResizedCrop.ACTIVE_SIZE
        img = F.resize(img, target_size, self.interpolation)
        return img

    def __repr__(self):
        return "MyResize(size=%s%s, interpolation=%s)" % (
            MyRandomResizedCrop.IMAGE_SIZE_LIST,
            "@continuous" if MyRandomResizedCrop.CONTINUOUS else "",
            _pil_interpolation_to_str[self.interpolation],
        )
