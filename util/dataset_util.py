import io
import json
import logging
import math
import os
import random
from typing import List

import requests
from PIL import Image, UnidentifiedImageError
from requests import HTTPError, Timeout, TooManyRedirects

# Setup logger
log = logging.getLogger("dataset_util")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", '%m/%d/%Y %I:%M:%S %p')
stream_h = logging.StreamHandler()
stream_h.setLevel(logging.DEBUG)
stream_h.setFormatter(formatter)
log.addHandler(stream_h)


class DatasetUtil:

    def __init__(self, base_dir: str = './datasets/', imagenet_dir: str = "tiny-imagenet-200",
                 utk_dir_name: str = "UTKFace", vgg_dataset_dir: str = "vgg_face_dataset",
                 total_class_count: int = 10, img_size: int = 64, train_img_count: int = 500,
                 validation_ratio: float = 0.1, vgg_download: bool = False, seed: int = 10):
        """
        Prepare tiny imagenet, utkface, vgg datasets
        :param base_dir: Datasets folder that contains both tiny-imagenet-200 and UTKFace folders
        :param imagenet_dir: Name of the Tiny Imagenet dataset directory
        :param utk_dir_name: Name of the UTKFace dataset directory
        :param vgg_dataset_dir: Name of the VGG dataset directory
        :param total_class_count: Total number of class to use.
        :param img_size: Target image size. Used to match image size with ImageNet (utk-util)
        :param train_img_count: Number of images to use for training per class
        :param validation_ratio: Validation dataset ratio
        :param seed: Seed for reproducibility.
        """
        random.seed(seed)

        # Common variables
        self.int2name = {}  # 7335 -> "goldfish, Carassius auratus"
        self._base_dir = base_dir
        self._train_img_count = train_img_count
        self._validation_ratio = validation_ratio
        self._img_size = img_size

        # Imagenet variables
        self.imagenet_train = []  # [("n01443537_0.jpeg", 7335), ...]
        self.imagenet_val = []  # [("n01443537_0.jpeg", 7335), ...]
        self.imagenet_id2int = {}  # n01443537 -> 7335
        self.imagenet_dir = os.path.join(base_dir, imagenet_dir)
        self._imagenet_cls_cnt = total_class_count - 1
        self._imagenet_id2name = {}  # n01443537 -> "goldfish, Carassius auratus"

        # UTKFace variables
        self.utk_train = []
        self.utk_val = []
        self.utk_post_proc_dir = os.path.join(self._base_dir, "modified_datasets", "utk")
        self._utk_dir = utk_dir_name
        self._utk_src = os.path.join(base_dir, utk_dir_name)
        self._utk_file_dict = {
            "male": {
                "white": [],
                "black": [],
                "asian": [],
                "indian": [],
                "other": []
            },
            "female": {
                "white": [],
                "black": [],
                "asian": [],
                "indian": [],
                "other": []
            }
        }

        # VGG variables
        self.vgg_train = []
        self.vgg_val = []
        self.vgg_post_proc_dir = os.path.join(self._base_dir, "modified_datasets", 'vgg')
        self._vgg_dir = os.path.join(base_dir, vgg_dataset_dir)

        # Create post-processing directories
        os.makedirs(self.utk_post_proc_dir, exist_ok=True)
        os.makedirs(self.vgg_post_proc_dir, exist_ok=True)

        # Load Tiny ImageNet datasets
        self._imagenet_load_words_txt()
        self._imagenet_load_images(self._imagenet_select_classes())

        # Load UTKFace datasets
        self._utk_load_images()
        self._utk_update_images()

        # Load vgg datasets
        if vgg_download:
            self.vgg_download_images()
        else:
            self._vgg_load_images()
            self._vgg_update_images()

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
                self.imagenet_train.append((file_n, cls))
                if i >= self._train_img_count:
                    break

        # Read validation index files
        with open(os.path.join(self.imagenet_dir, "val", "val_annotations.txt"), "r", encoding="utf-8") as f:
            for line in f:
                ln = str(line).strip().split()
                file_n = ln[0]
                cls_id = ln[1]
                if cls_id in class_list:
                    self.imagenet_val.append((file_n, cls_id))

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
            elif sep[2] == "4":
                race = "other"
            else:
                continue

            self._utk_file_dict[gender][race].append((file_name, len(self.int2name) - 1))

        genders = ["male", "female"]
        races = ["white", "black", "asian", "indian", "other"]

        # Equally distribute all classes to generate fair training/validation data
        img_per_cls = math.ceil(self._train_img_count / (len(genders) * len(races)))
        val_per_cls = math.ceil(img_per_cls * self._validation_ratio)
        log.debug("Images to get per class: %s" % img_per_cls)
        for gen in genders:
            for rac in races:
                log.debug("Current gender: %s\tCurrent race: %s\tCurrent class size:%s" % (
                    gen, rac, len(self._utk_file_dict[gen][rac])))
                # Shuffle data
                random.shuffle(self._utk_file_dict[gen][rac])
                # Create unique training dataset
                self.utk_train.extend(self._utk_file_dict[gen][rac][:img_per_cls])
                # Create unique validation dataset
                self.utk_val.extend(
                    self._utk_file_dict[gen][rac][img_per_cls:img_per_cls + val_per_cls])
        log.debug("Total length of training dataset: %s" % len(self.utk_train))
        log.debug("Total length of validation dataset: %s" % len(self.utk_val))

    def _utk_update_images(self):
        # Create modified training data directory
        train_dir = os.path.join(self.utk_post_proc_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # Create modified validation data directory
        val_dir = os.path.join(self.utk_post_proc_dir, "val")
        os.makedirs(val_dir, exist_ok=True)

        # Resize all training dataset images
        for img_n, _ in self.utk_train:
            with Image.open(os.path.join(self._utk_src, img_n)) as img:
                img = img.resize((self._img_size, self._img_size))
                img.save(os.path.join(train_dir, img_n))
        log.debug("Resized training images. Saved at %s" % train_dir)

        # Resize all validation dataset images
        for img_n, _ in self.utk_val:
            with Image.open(os.path.join(self._utk_src, img_n)) as img:
                img = img.resize((self._img_size, self._img_size))
                img.save(os.path.join(val_dir, img_n))
        log.debug("Resized validation images. Saved at %s\n" % val_dir)

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
    #         log.debug("UTKFace validation list loaded from: %s" % val_f_n)
    #     else:
    #         train_f_n = os.path.join(self._utk_dir, "train.json")
    #         with open(train_f_n, "r", encoding="utf-8") as f:
    #             self._utk_dataset = json.load(f)
    #         log.debug("UTKFace training list loaded from: %s" % train_f_n)
    #
    #     for data in self._utk_dataset:
    #         # ('9_0_0_20170110220232058.jpg.chip.jpg', 9)
    #         self._images.append((data, len(self._int2name) - 1))

    # # saves utk file lists
    # def utk_save_to_file(self) -> None:
    #     train_f_n = os.path.join(self.save_dir, "train.json")
    #     with open(train_f_n, "w", encoding="utf-8") as f:
    #         json.dump(self._training_dataset, f)
    #     log.debug("Training list saved as: %s" % train_f_n)
    #
    #     val_f_n = os.path.join(self.save_dir, "validation.json")
    #     with open(val_f_n, "w", encoding="utf-8") as f:
    #         json.dump(self._validation_dataset, f)
    #     log.debug("Validation list saved as: %s\n" % val_f_n)

    def vgg_download_images(self) -> None:
        curr_downloaded = 0
        save_dir = os.path.join(self._base_dir, "modified_datasets", "vgg", "downloaded")
        os.makedirs(save_dir, exist_ok=True)
        allowed_img_ext = [".jpg", ".jpeg", ".png"]
        files = os.listdir(os.path.join(self._vgg_dir, "files"))
        log.debug("File list: %s" % str(files))
        # Use random files
        random.shuffle(files)
        # download another validation size images to download extra images and remove some invalid images
        total = self._train_img_count + self._train_img_count * self._validation_ratio * 2
        while curr_downloaded < total:
            if curr_downloaded % 10 == 0:
                log.info("Current progress: %s/%s" % (curr_downloaded, total))
            file_n = files[curr_downloaded]
            if file_n.endswith(".txt"):
                downloaded = False
                person_n = file_n[:-4]
                # log.debug("Current file: %s" % file_n)
                with open(os.path.join(self._vgg_dir, "files", file_n), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    while not downloaded:
                        ln_idx = random.randint(0, len(lines) - 1)
                        ln = lines[ln_idx].strip().split()
                        url = ln[1]
                        img_ext = url[-4:]
                        left = math.floor(float(ln[2]))
                        top = math.floor(float(ln[3]))
                        right = math.floor(float(ln[4]))
                        bot = math.floor(float(ln[5]))
                        # If left, top data of the bounding box is bigger than 0 and
                        # image extension is known, and if the bounding box size is bigger than the given size
                        if (0 <= left and 0 <= top) and img_ext.lower() in allowed_img_ext and (
                                self._img_size <= right - left and self._img_size <= bot - top):
                            img_name = person_n + "-" + ln[0] + img_ext
                            log.debug("%s" % img_name)
                            try:
                                with requests.get(url, timeout=5) as res:
                                    log.debug("Response code: %s\tSize:%s" % (str(res.status_code),
                                                                              str(len(res.content))))
                                    # If status code is normal and content size is not 0
                                    if res.status_code == 200 and len(res.content) > 1000:
                                        try:
                                            with Image.open(io.BytesIO(res.content)).convert('RGB') as img:
                                                img = img.crop((left, top, right, bot))
                                                img_path = os.path.join(save_dir, img_name)
                                                img.save(img_path)
                                                if os.path.getsize(img_path) > 1100:
                                                    downloaded = True
                                                    curr_downloaded += 1
                                                else:
                                                    log.debug("File too small.")
                                                    os.remove(img_path)
                                        except UnidentifiedImageError:
                                            log.debug("Not an image file?")
                                            continue
                            except ConnectionError:
                                log.debug("ConnectionError")
                                continue
                            except HTTPError:
                                log.debug("HTTPError")
                                continue
                            except Timeout:
                                log.debug("Timeout")
                                continue
                            except TooManyRedirects:
                                log.debug("TooManyRedirects")
                                continue
                            except Exception as err:
                                log.debug(err)
                                continue
        self._vgg_load_images()
        self._vgg_update_images()

    def _vgg_load_images(self) -> None:
        _images = []
        for file_n in os.listdir(os.path.join(self.vgg_post_proc_dir, 'downloaded')):
            _images.append((file_n, len(self.int2name) - 1))
        random.shuffle(_images)
        val_size = math.ceil(self._train_img_count * self._validation_ratio)
        self._training = _images[:self._train_img_count]
        self._validating = _images[self._train_img_count:self._train_img_count + val_size]
        log.debug("Length of vgg training images: %s" % len(self._training))
        log.debug("Length of vgg validating images: %s" % len(self._validating))

    def _vgg_update_images(self) -> None:
        # create modified validation data directory
        val_dir = os.path.join(self.vgg_post_proc_dir, "val")
        os.makedirs(val_dir, exist_ok=True)

        # create modified training data directory
        train_dir = os.path.join(self.vgg_post_proc_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # Resize all training dataset images
        for img_n, _ in self._training:
            with Image.open(os.path.join(self.vgg_post_proc_dir, 'downloaded', img_n)) as img:
                img = img.resize((self._img_size, self._img_size))
                img.save(os.path.join(train_dir, img_n))

        # Resize all validation dataset images
        for img_n, _ in self._validating:
            with Image.open(os.path.join(self.vgg_post_proc_dir, 'downloaded', img_n)) as img:
                img = img.resize((self._img_size, self._img_size))
                img.save(os.path.join(val_dir, img_n))
        log.info("Resized images saved.")

    def vgg_save_to_file(self) -> None:
        val_f_n = os.path.join(self.vgg_post_proc_dir, "validation.json")
        with open(val_f_n, "w", encoding="utf-8") as f:
            json.dump(self._validating, f)
        log.info("Validation list saved as: %s" % val_f_n)

        train_f_n = os.path.join(self.vgg_post_proc_dir, "train.json")
        with open(train_f_n, "w", encoding="utf-8") as f:
            json.dump(self._training, f)
        log.info("Training list saved as: %s" % train_f_n)


# Just for testing
if __name__ == '__main__':
    test = DatasetUtil(base_dir="../datasets/")
    # test.vgg_download_images()
    print(test.imagenet_train[:5])
    print(test.imagenet_val[:5])
    print(test.utk_train[:5])
    print(test.utk_val[:5])
