import json
import os
import random
from typing import List

import PIL.Image
import numpy as np
import torch.utils.data
import torchvision.transforms as vision

from util.utk_util import UTKFaceUtil


class ImagenetUtk(torch.utils.data.Dataset):
    def __init__(self, base_dir: str = '../datasets/', imagenet_dir: str = "tiny-imagenet-200", image_size: int = 224,
                 total_class_count: int = 10, validation: bool = False, seed: int = 10, use_utk_util: bool = True,
                 utk_dir_name: str = "UTKFace", utk_img_size: int = 64, training_size: int = 500,
                 validation_ratio: float = 0.1, save_to_file: bool = True):
        """
        Initialize the Imagenet + UTKFace dataset
        "(utk-util)" in comments below indicate parameters that are used by utk-util.py. Some parameters are shared.
        :param base_dir: Datasets folder that contains both tiny-imagenet-200 and UTKFace folders
        :param imagenet_dir: Name of the Tiny Imagenet dataset directory
        :param image_size: Target image size (A method will add paddings or trim the image in the memory)
        :param total_class_count: Total number of class to use.
        :param validation: True to use validation data, False otherwise.
        :param seed: Seed for reproducibility.
        :param use_utk_util: True to use utk_util with default settings, False otherwise. (utk-util)
        :param utk_dir_name: Name of the UTKFace dataset directory (utk-util)
        :param utk_img_size: Target image size for utk_img_size. Used to match image size with ImageNet (utk-util)
        :param training_size: Training data size for UTKFace dataset (utk-util)
        :param validation_ratio: Validation dataset ratio for UTKFace dataset (utk-util)
        :param save_to_file: True to save UTKFace location files, False otherwise. (utk-util)
        """
        random.seed(seed)
        if use_utk_util:
            utk_util = UTKFaceUtil(base_dir=base_dir, utk_dir_name=utk_dir_name, img_seed=seed,
                                   training_size=training_size, validation_ratio=validation_ratio,
                                   img_size=utk_img_size, save_to_file=False)
            # To suppress a warning :)
            if save_to_file:
                utk_util.save_to_file()

        self._images = []  # [("n01443537_0.jpeg", 7335), ...]
        self._id2name = {}  # n01443537 -> "goldfish, Carassius auratus"
        self._id2int = {}  # n01443537 -> 7335
        self._int2name = {}  # 7335 -> "goldfish, Carassius auratus"

        self._image_size = image_size
        self._base_dir = base_dir
        self._imagenet_dir = os.path.join(base_dir, imagenet_dir)
        self._utk_dir = os.path.join(self._base_dir, "modified_datasets", "utk")
        self._imagenet_cls_cnt = total_class_count - 1
        self._load_images(self._load_imagenet_words_txt())
        self._utk_validation = validation

        self._load_utk_images(validation)

    def __getitem__(self, idx) -> (torch.Tensor, int):
        """
        Reads the RGB image file at idx, and return it as an image tensor with an int label.
        Int label can be converted to a human-readable label with get_class_name() function.
        :param idx: Index of data to get
        :return: Tensor containing image as an array, and an integer representing the image number
        """
        img_name = self._images[idx][0]

        # if image at idx belongs in imagenet
        if str(img_name).startswith("n"):
            cls_id = self._images[idx][1]
            cls_int = self._id2int[cls_id]
            full_file_name = os.path.join(self._imagenet_dir, "train", cls_id, "images", img_name)
        else:
            cls_int = self._images[idx][1]
            if self._utk_validation:
                full_file_name = os.path.join(self._utk_dir, "val", img_name)
            else:
                full_file_name = os.path.join(self._utk_dir, "train", img_name)

        # Uncomment the following line for absolute path
        # full_file_name = os.path.abspath(full_file_name)
        image = PIL.Image.open(full_file_name).convert("RGB")

        # Add paddings around the image
        crop = vision.CenterCrop(self._image_size)
        arr = np.transpose(np.array(crop(image)), (2, 0, 1))
        return torch.from_numpy(arr).float(), cls_int

    def __len__(self):
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
        if int_label == 9:
            return "face"
        else:
            return self._int2name[int_label]

    def _load_imagenet_words_txt(self) -> List[str]:
        """
        Using words.txt in Tiny ImageNet to create index file, then select several classes
        :return: List of Tiny ImageNet classes ID to use
        """
        # First, create word index file
        with open(os.path.join(self._imagenet_dir, "words.txt"), "r", encoding="utf-8") as f:
            for line in f:
                ln = str(line).strip().split("\t", maxsplit=2)
                cls_id = ln[0]
                cls_name = ln[1]
                self._id2name[cls_id] = cls_name

        # Shuffle class order
        f_names = os.listdir(os.path.join(self._imagenet_dir, "train"))
        random.shuffle(f_names)

        cls_names = []
        i = 0

        # Only get few (self._imagenet_cls_cnt) classes to save time during training
        while len(cls_names) < self._imagenet_cls_cnt:
            f_name = f_names[i]
            # Double check if folder name exist in the dictionary
            if f_name in self._id2name:
                # Create new index dictionary
                self._id2int[f_name] = len(self._id2int)
                self._int2name[len(self._int2name)] = self._id2name[f_name]
                cls_names.append(f_name)
            i += 1
        return cls_names

    def _load_images(self, class_list: List[str]) -> None:
        """
        Load images to dataset list
        :param class_list: List of ImageNet class ID to include
        :return: None
        """
        for cls in class_list:
            for file_n in os.listdir(os.path.join(self._imagenet_dir, "train", cls, "images")):
                # ('n02950826_81.JPEG', 'n02950826')
                self._images.append((file_n, cls))

    def _load_utk_images(self, validation: bool) -> None:
        """
        Load UTKFace dataset that was created with utk_util.py, and append to dataset list
        :param validation: True if using UTKFace validation dataset, False otherwise
        :return: None
        """
        if validation:
            val_f_n = os.path.join(self._utk_dir, "validation.json")
            with open(val_f_n, "r", encoding="utf-8") as f:
                self._utk_dataset = json.load(f)
            print("UTKFace validation list loaded from: %s" % val_f_n)
        else:
            train_f_n = os.path.join(self._utk_dir, "train.json")
            with open(train_f_n, "r", encoding="utf-8") as f:
                self._utk_dataset = json.load(f)
            print("UTKFace training list loaded from: %s" % train_f_n)

        # Hard code "face" label
        self._int2name[len(self._int2name)] = "face"
        for data in self._utk_dataset:
            # ('9_0_0_20170110220232058.jpg.chip.jpg', 9)
            self._images.append((data, len(self._int2name) - 1))


# Just for testing
if __name__ == '__main__':
    test = ImagenetUtk(image_size=64)
    # TinyImagenet dataset
    print("Shape: %s\tLabel: %s" % (test[0][0].shape, test[0][1]))
    # UTKFace dataset
    print("Shape: %s\tLabel: %s" % (test[5000][0].shape, test[5000][1]))
