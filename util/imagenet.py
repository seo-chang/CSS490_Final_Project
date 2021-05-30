import os
from typing import List

import PIL.Image
import numpy as np
import torch.utils.data
import torchvision.transforms as vision

from util.dataset_util import DatasetUtil


# import logging as log


class Imagenet(torch.utils.data.Dataset):

    def __init__(self, du: DatasetUtil, base_dir: str = './datasets', image_size: int = 224, validation: bool = False):
        """
        Initialize the Imagenet dataset
        :param du: Dataset utility class
        :param base_dir: Datasets folder that contains Imagenet-200
        :param image_size:
        :param validation:True to use validation data, False otherwise
        """
        self.base_dir = base_dir
        self._du = du

        self._images = []
        self._image_size = image_size
        self._validation = validation

        if not validation:
            self._images.extend(du.imagenet_train)
        else:
            self._images.extend(du.imagenet_val)

    def __getitem__(self, idx) -> (torch.Tensor, int):
        """
        Reads the RGB image file at idx, and return it as an image tensor with an int label.
        Int label can be converted to a human-readable label with get_class_name() function.
        :param idx: Index of data to get
        :return: Tensor containing image as an array, and an integer representing the image number
        """
        img_name = self._images[idx][0]
        cls_id = self._images[idx][1]
        cls_int = self._du.imagenet_id2int[cls_id]
        if not self._validation:
            full_file_name = os.path.join(self._du.imagenet_dir, "train", cls_id, "images", img_name)
        else:
            full_file_name = os.path.join(self._du.imagenet_dir, "val", "images", img_name)

        # Uncomment the following line for absolute path
        # full_file_name = os.path.abspath(full_file_name)
        image = PIL.Image.open(full_file_name).convert("RGB")

        # Add paddings around the image
        crop = vision.CenterCrop(self._image_size)
        # TODO: double check if uint8 can be used. until then, use float as dtype
        # arr = np.transpose(np.array(crop(image), dtype="uint8"), (2, 0, 1))
        # return torch.from_numpy(arr), cls_int
        arr = np.transpose(np.array(crop(image)), (2, 0, 1))
        return torch.from_numpy(arr).float(), cls_int

    def __len__(self) -> int:
        """
        Get length of the dataset
        :return: Total image count
        """
        return len(self._images)

    def get_class_name(self, int_label: int) -> str:
        """
        Convert an integer label to a human-readable class name
        :param int_label: Integer label to convert
        :return: Human-readable class name. e.g: "cat", "face", etc..
        """
        return self._du.int2name[int_label]

    def get_class_names(self) -> List[str]:
        """
        Returns list of class names
        :return: List of possible class names
        """
        return list(self._du.int2name.values())


if __name__ == '__main__':
    data_util = DatasetUtil(base_dir="../datasets/")
    test = Imagenet(data_util, image_size=64)
    cls_list = test.get_class_names()
    print("IMAGENET: total class size: %s" % len(cls_list))
    print("IMAGENET: total dataset size: %s" % len(test))

    print("IMAGENET: torch tensor type: %s" % test[0][0].dtype)
    print("IMAGENET: shape: %s\tLabel: %s\tString label: %s" % (
        test[0][0].shape, test[0][1], test.get_class_name(test[0][1])))
    img = test.__getitem__(500)
    print(img)
