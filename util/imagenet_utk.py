import json
import math
import os
import random
from typing import List

import PIL.Image
import numpy as np
import torch.utils.data
import torchvision.transforms as vision


class ImagenetUtk(torch.utils.data.Dataset):
    def __init__(self, base_dir: str = '../datasets/', imagenet_dir: str = "tiny-imagenet-200", image_size: int = 224,
                 total_class_count: int = 10, validation: bool = False, seed: int = 10, initialize: bool = True,
                 use_utk_util: bool = True, utk_dir_name: str = "UTKFace", utk_img_size: int = 64,
                 training_size: int = 500, validation_ratio: float = 0.1, save_to_file: bool = True):
        """
        Initialize the Imagenet + UTKFace dataset
        "(utk-util)" in comments below indicate parameters that are used by utk-util.py. Some parameters are shared.
        :param base_dir: Datasets folder that contains both tiny-imagenet-200 and UTKFace folders
        :param imagenet_dir: Name of the Tiny Imagenet dataset directory
        :param image_size: Target image size (A method will add paddings or trim the image in the memory)
        :param total_class_count: Total number of class to use.
        :param validation: True to use validation data, False otherwise.
        :param seed: Seed for reproducibility.
        :param initialize: True to use utk_util with default settings, False otherwise. (utk-util)
        :param use_utk_util: Alias of initialize parameter. (backward compatibility) (utk-util)
        :param utk_dir_name: Name of the UTKFace dataset directory (utk-util)
        :param utk_img_size: Target image size for utk_img_size. Used to match image size with ImageNet (utk-util)
        :param training_size: Training data size for UTKFace dataset (utk-util)
        :param validation_ratio: Validation dataset ratio for UTKFace dataset (utk-util)
        :param save_to_file: True to save UTKFace location files, False otherwise. (utk-util)
        """
        random.seed(seed)
        if initialize or use_utk_util:
            utk_util = _UTKFaceUtil(base_dir=base_dir, utk_dir_name=utk_dir_name, img_seed=seed,
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
        arr = np.transpose(np.array(crop(image), dtype="uint8"), (2, 0, 1))
        return torch.from_numpy(arr), cls_int

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

    def get_class_names(self) -> List[str]:
        """
        Returns list of class names
        :return: List of possible class names
        """
        return list(self._int2name.values())

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
        Load UTKFace dataset that was created with UTKFaceUtil, and append to dataset list
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


class _UTKFaceUtil:
    def __init__(self, base_dir: str = '../datasets/', utk_dir_name: str = "UTKFace", img_seed: int = 10,
                 training_size: int = 500, validation_ratio: float = 0.1, img_size: int = 64,
                 save_to_file: bool = True):
        """
        Creates unique training/validating datasets for UTKFace dataset and
        :param base_dir: Datasets folder that contains both tiny-imagenet-200 and UTKFace folders
        :param utk_dir_name: UTKFace dataset directory name
        :param img_seed: Seed for random number generator
        :param training_size: Target training dataset size
        :param validation_ratio: Validation dataset ratio
        :param img_size: Target image size (this value will be used for resizing)
        :param save_to_file: True to save to files, False otherwise
        """
        random.seed(img_seed)

        self._base_dir = base_dir
        self._utk_dir = utk_dir_name
        self._dataset_size = training_size
        self._validation_ratio = validation_ratio
        self._src = os.path.join(base_dir, utk_dir_name)
        self._img_size = img_size
        self.save_dir = os.path.join(self._base_dir, "modified_datasets", "utk")
        os.makedirs(self.save_dir, exist_ok=True)

        self._training_dataset = []
        self._validation_dataset = []
        self._file_dict = {
            "male": {
                "white": [],
                "black": [],
                "asian": [],
                "indian": []
            },
            "female": {
                "white": [],
                "black": [],
                "asian": [],
                "indian": []
            }
        }
        self._load_images()
        self._update_img()

        if save_to_file:
            self.save_to_file()

    def save_to_file(self) -> None:
        train_f_n = os.path.join(self.save_dir, "train.json")
        with open(train_f_n, "w", encoding="utf-8") as f:
            json.dump(self._training_dataset, f)
        print("Training list saved as: %s" % train_f_n)

        val_f_n = os.path.join(self.save_dir, "validation.json")
        with open(val_f_n, "w", encoding="utf-8") as f:
            json.dump(self._validation_dataset, f)
        print("Validation list saved as: %s\n" % val_f_n)

    def _load_images(self) -> None:
        """
        Iterate through images and parse data
        :return: None
        """
        for file_name in os.listdir(self._src):
            sep = file_name.split('_')

            # Parse gender
            if sep[1] == "0":
                gender = "male"
            elif sep[1] == "1":
                gender = "female"
            else:
                continue

            # Parse race
            if sep[2] == "0":
                race = "white"
            elif sep[2] == "1":
                race = "black"
            elif sep[2] == "3":
                race = "asian"
            elif sep[2] == "4":
                race = "indian"
            else:
                continue

            self._file_dict[gender][race].append(file_name)

        genders = ["male", "female"]
        races = ["white", "black", "asian", "indian"]

        # Equally distribute all classes to generate fair training/validation data
        img_per_cls = math.ceil(self._dataset_size / (len(genders) * len(races)))
        val_per_cls = math.ceil(img_per_cls * self._validation_ratio)
        print("Images to get per class: %s" % img_per_cls)
        for gen in genders:
            for rac in races:
                print("Current gender: %s\tCurrent race: %s Current class size:%s" % (
                    gen, rac, len(self._file_dict[gen][rac])))
                # Shuffle data
                random.shuffle(self._file_dict[gen][rac])
                # Create unique training dataset
                self._training_dataset.extend(self._file_dict[gen][rac][:img_per_cls])
                # Create unique validation dataset
                self._validation_dataset.extend(self._file_dict[gen][rac][img_per_cls:img_per_cls + val_per_cls])
        print("Total length of training dataset: %s" % len(self._training_dataset))
        print("Total length of validation dataset: %s\n" % len(self._validation_dataset))

    def _update_img(self):
        # Create modified training data directory
        train_dir = os.path.join(self.save_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # Create modified validation data directory
        val_dir = os.path.join(self.save_dir, "val")
        os.makedirs(val_dir, exist_ok=True)

        # Resize all training dataset images
        for img_n in self._training_dataset:
            with PIL.Image.open(os.path.join(self._src, img_n)) as img:
                img = img.resize((self._img_size, self._img_size))
                img.save(os.path.join(train_dir, img_n))
        print("Resized training images. Saved at %s" % train_dir)

        # Resize all validation dataset images
        for img_n in self._validation_dataset:
            with PIL.Image.open(os.path.join(self._src, img_n)) as img:
                img = img.resize((self._img_size, self._img_size))
                img.save(os.path.join(val_dir, img_n))
        print("Resized validation images. Saved at %s\n" % val_dir)


# Just for testing
if __name__ == '__main__':
    test = ImagenetUtk(image_size=64)
    cls_list = test.get_class_names()
    print("Total class size: %s" % len(cls_list))
    print("Torch tensor type: %s" % test[0][0].dtype)
    # TinyImagenet dataset
    print("Shape: %s\tLabel: %s" % (test[0][0].shape, test[0][1]))
    # UTKFace dataset
    print("Shape: %s\tLabel: %s" % (test[5000][0].shape, test[5000][1]))
