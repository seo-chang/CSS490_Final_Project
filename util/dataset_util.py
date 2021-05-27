import math
import os
import random
from typing import List

import PIL.Image


class DatasetUtil:

    def __init__(self, base_dir: str = './datasets/', imagenet_dir: str = "tiny-imagenet-200",
                 utk_dir_name: str = "UTKFace", total_class_count: int = 10, utk_img_size: int = 64,
                 train_img_count: int = 500, validation_ratio: float = 0.1, seed: int = 10):
        """
        Prepare tiny imagenet, utkface, vgg datasets
        :param base_dir: Datasets folder that contains both tiny-imagenet-200 and UTKFace folders
        :param imagenet_dir: Name of the Tiny Imagenet dataset directory
        :param utk_dir_name: Name of the UTKFace dataset directory
        :param total_class_count: Total number of class to use.
        :param utk_img_size: Target image size for utk_img_size. Used to match image size with ImageNet (utk-util)
        :param train_img_count: Number of images to use for training per class
        :param validation_ratio: Validation dataset ratio
        :param seed: Seed for reproducibility.
        """
        random.seed(seed)

        # Common variables
        self.int2name = {}  # 7335 -> "goldfish, Carassius auratus"
        self._base_dir = base_dir
        self._train_img_count = train_img_count

        # Imagenet variables
        self.imagenet_train_images = []  # [("n01443537_0.jpeg", 7335), ...]
        self.imagenet_val_images = []  # [("n01443537_0.jpeg", 7335), ...]
        self.imagenet_id2int = {}  # n01443537 -> 7335
        self.imagenet_dir = os.path.join(base_dir, imagenet_dir)
        self._imagenet_cls_cnt = total_class_count - 1
        self._imagenet_id2name = {}  # n01443537 -> "goldfish, Carassius auratus"

        # UTKFace variables
        self.utk_training_dataset = []
        self.utk_val_dataset = []
        self.utk_save_dir = os.path.join(self._base_dir, "modified_datasets", "utk")
        self._utk_dir = utk_dir_name
        self._utk_src = os.path.join(base_dir, utk_dir_name)
        self._utk_validation_ratio = validation_ratio
        self._utk_img_size = utk_img_size
        self._utk_file_dict = {
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

        os.makedirs(self.utk_save_dir, exist_ok=True)

        # Load Tiny ImageNet datasets
        self._imagenet_load_words_txt()
        self._imagenet_load_images(self._imagenet_select_classes())

        # Load UTKFace datasets
        self._utk_load_images()
        self._utk_update_img()

    def _imagenet_load_words_txt(self) -> None:
        """
        Using words.txt in Tiny ImageNet to create index file, then select several classes
        :return: None
        """
        # create word index file
        with open(os.path.join(self.imagenet_dir, "words.txt"), "r", encoding="utf-8") as f:
            for line in f:
                ln = str(line).strip().split("\t", maxsplit=2)
                cls_id = ln[0]
                cls_name = ln[1]
                self._imagenet_id2name[cls_id] = cls_name

    def _imagenet_select_classes(self) -> List[str]:
        # shuffle class order
        f_names = os.listdir(os.path.join(self.imagenet_dir, "train"))
        random.shuffle(f_names)

        i = 0
        cls_names = []
        # select a few (self._imagenet_cls_cnt) classes to save time during training
        while len(cls_names) < self._imagenet_cls_cnt:
            f_name = f_names[i]
            # Double check if folder name exist in the dictionary
            if f_name in self._imagenet_id2name:
                # Create new index dictionary
                self.imagenet_id2int[f_name] = len(self.imagenet_id2int)
                self.int2name[len(self.int2name)] = self._imagenet_id2name[f_name]
                cls_names.append(f_name)
            i += 1
        # Hard code "face" label
        self.int2name[len(self.int2name)] = "face"
        return cls_names

    def _imagenet_load_images(self, class_list: List[str]) -> None:
        """
        Load training/validating images to respective dataset list
        :param class_list: List of ImageNet class ID to include
        :return: None
        """
        for cls in class_list:
            i = 0
            for file_n in os.listdir(os.path.join(self.imagenet_dir, "train", cls, "images")):
                # ('n02950826_81.JPEG', 'n02950826')
                self.imagenet_train_images.append((file_n, cls))
                if i >= self._train_img_count:
                    break

        # Read validation index files
        with open(os.path.join(self.imagenet_dir, "val", "val_annotations.txt"), "r", encoding="utf-8") as f:
            for line in f:
                ln = str(line).strip().split()
                file_n = ln[0]
                cls_id = ln[1]
                if cls_id in class_list:
                    self.imagenet_val_images.append((file_n, cls_id))

    def _utk_load_images(self) -> None:
        """
        Iterate through images and parse data
        :return: None
        """
        for file_name in os.listdir(self._utk_src):
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
            elif sep[2] == "2":
                race = "asian"
            elif sep[2] == "3":
                race = "indian"
            else:
                continue

            self._utk_file_dict[gender][race].append((file_name, len(self.int2name) - 1))

        genders = ["male", "female"]
        races = ["white", "black", "asian", "indian"]

        # Equally distribute all classes to generate fair training/validation data
        img_per_cls = math.ceil(self._train_img_count / (len(genders) * len(races)))
        val_per_cls = math.ceil(img_per_cls * self._utk_validation_ratio)
        print("Images to get per class: %s" % img_per_cls)
        for gen in genders:
            for rac in races:
                print("Current gender: %s\tCurrent race: %s Current class size:%s" % (
                    gen, rac, len(self._utk_file_dict[gen][rac])))
                # Shuffle data
                random.shuffle(self._utk_file_dict[gen][rac])
                # Create unique training dataset
                self.utk_training_dataset.extend(self._utk_file_dict[gen][rac][:img_per_cls])
                # Create unique validation dataset
                self.utk_val_dataset.extend(
                    self._utk_file_dict[gen][rac][img_per_cls:img_per_cls + val_per_cls])
        print("Total length of training dataset: %s" % len(self.utk_training_dataset))
        print("Total length of validation dataset: %s\n" % len(self.utk_val_dataset))

    def _utk_update_img(self):
        # Create modified training data directory
        train_dir = os.path.join(self.utk_save_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # Create modified validation data directory
        val_dir = os.path.join(self.utk_save_dir, "val")
        os.makedirs(val_dir, exist_ok=True)

        # Resize all training dataset images
        for img_n, _ in self.utk_training_dataset:
            with PIL.Image.open(os.path.join(self._utk_src, img_n)) as img:
                img = img.resize((self._utk_img_size, self._utk_img_size))
                img.save(os.path.join(train_dir, img_n))
        print("Resized training images. Saved at %s" % train_dir)

        # Resize all validation dataset images
        for img_n, _ in self.utk_val_dataset:
            with PIL.Image.open(os.path.join(self._utk_src, img_n)) as img:
                img = img.resize((self._utk_img_size, self._utk_img_size))
                img.save(os.path.join(val_dir, img_n))
        print("Resized validation images. Saved at %s\n" % val_dir)

    # # loads utk file lists
    # def _utk_load_images(self, validation: bool) -> None:
    #     """
    #     Load UTKFace dataset that was created with UTKFaceUtil, and append to dataset list
    #     :param validation: True if using UTKFace validation dataset, False otherwise
    #     :return: None
    #     """
    #     if validation:
    #         val_f_n = os.path.join(self._utk_dir, "validation.json")
    #         with open(val_f_n, "r", encoding="utf-8") as f:
    #             self._utk_dataset = json.load(f)
    #         print("UTKFace validation list loaded from: %s" % val_f_n)
    #     else:
    #         train_f_n = os.path.join(self._utk_dir, "train.json")
    #         with open(train_f_n, "r", encoding="utf-8") as f:
    #             self._utk_dataset = json.load(f)
    #         print("UTKFace training list loaded from: %s" % train_f_n)
    #
    #     for data in self._utk_dataset:
    #         # ('9_0_0_20170110220232058.jpg.chip.jpg', 9)
    #         self._images.append((data, len(self._int2name) - 1))

    # # saves utk file lists
    # def utk_save_to_file(self) -> None:
    #     train_f_n = os.path.join(self.save_dir, "train.json")
    #     with open(train_f_n, "w", encoding="utf-8") as f:
    #         json.dump(self._training_dataset, f)
    #     print("Training list saved as: %s" % train_f_n)
    #
    #     val_f_n = os.path.join(self.save_dir, "validation.json")
    #     with open(val_f_n, "w", encoding="utf-8") as f:
    #         json.dump(self._validation_dataset, f)
    #     print("Validation list saved as: %s\n" % val_f_n)


# Just for testing
if __name__ == '__main__':
    test = DatasetUtil(base_dir="../datasets/")
    print(test.imagenet_train_images[:5])
    print(test.imagenet_val_images[:5])
    print(test.utk_training_dataset[:5])
    print(test.utk_val_dataset[:5])
