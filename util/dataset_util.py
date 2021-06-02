import io
import json
import logging
import math
import os
import random
import shutil
from typing import List

import requests
from PIL import Image, UnidentifiedImageError
from requests import HTTPError, Timeout, TooManyRedirects

# Setup logger
log = logging.getLogger("dataset_util")
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", '%m/%d/%Y %I:%M:%S %p')
stream_h = logging.StreamHandler()
stream_h.setLevel(logging.DEBUG)
stream_h.setFormatter(formatter)
log.addHandler(stream_h)


# noinspection DuplicatedCode
class DatasetUtil:

    def __init__(self, base_dir: str = './datasets/', imagenet_dir: str = "tiny-imagenet-200",
                 utk_dir_name: str = "UTKFace", vgg_dataset_dir: str = "vgg_face_dataset",
                 total_class_count: int = 10, img_size: int = 64, train_img_count: int = 500,
                 validation_ratio: float = 0.1, vgg_download: bool = False, load_from_json: bool = False,
                 seed: int = 10):
        """
        Prepare tiny imagenet, utkface, vgg datasets
        :param base_dir: Datasets folder that contains both tiny-imagenet-200 and UTKFace folders
        :param imagenet_dir: Name of the Tiny Imagenet dataset directory
        :param utk_dir_name: Name of the UTKFace dataset directory
        :param vgg_dataset_dir: Name of the VGG dataset directory
        :param total_class_count: Total number of class to use.
        :param img_size: Target image size. Used to match utk, vgg image size with Tiny ImageNet images
        :param train_img_count: Number of images to use for training per class
        :param validation_ratio: Validation dataset ratio
        :param load_from_json: Load dataset lists from json files.
        :param seed: Seed for reproducibility.
        """
        random.seed(seed)

        # Common variables
        self.int2name = {}  # 7335 -> "goldfish, Carassius auratus"
        self._base_dir = base_dir
        self._post_proc_dir = os.path.join(self._base_dir, "modified_datasets")
        self._train_img_count = train_img_count
        self._validation_ratio = validation_ratio
        self._img_size = img_size

        # Imagenet variables
        self.imagenet_train = []  # [("n01443537_0.jpeg", 7335), ...]
        self.imagenet_val = []  # [("n01443537_0.jpeg", 7335), ...]
        self.imagenet_id2int = {}  # n01443537 -> 7335
        self.imagenet_dir = os.path.join(base_dir, imagenet_dir)
        self._imagenet_post_proc_dir = os.path.join(self._post_proc_dir, "imagenet")
        self._imagenet_cls_cnt = total_class_count - 1
        self._imagenet_id2name = {}  # n01443537 -> "goldfish, Carassius auratus"

        # UTKFace variables
        self.utk_train = []
        self.utk_val = []
        self.utk_post_proc_dir = os.path.join(self._post_proc_dir, "utk")
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
        self.vgg_post_proc_dir = os.path.join(self._post_proc_dir, 'vgg')
        self._vgg_dir = os.path.join(base_dir, vgg_dataset_dir)

        # VGG and UTK variables
        self.vgg_utk_val = []
        self.vgg_post_proc_dir = os.path.join(self._post_proc_dir, 'vgg')
        self._vgg_utk_val_dir = os.path.join(base_dir, 'modified_datasets', 'vgg_utk_val')

        # Create post-processing directories
        os.makedirs(self._imagenet_post_proc_dir, exist_ok=True)
        os.makedirs(self.utk_post_proc_dir, exist_ok=True)
        os.makedirs(self.vgg_post_proc_dir, exist_ok=True)
        os.makedirs(self._vgg_utk_val_dir, exist_ok=True)

        if load_from_json:
            self._load_index_json()
            self._imagenet_load_images_json()
            self._utk_load_images_json()
            self._vgg_load_images_json()
            self._vgg_utk_load_images_json()
        else:
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

            self._load_vgg_utk_val()

    def _save_index_json(self) -> None:
        # Save int2name data
        file_n = os.path.join(self._post_proc_dir, "int2name.json")
        with open(file_n, "w", encoding="utf-8") as f:
            json.dump(self.int2name, f)
        log.info("int2name saved as: %s" % file_n)

    def _load_index_json(self) -> None:
        # Load int2name
        file_n = os.path.join(self._post_proc_dir, "int2name.json")
        with open(file_n, "r", encoding="utf-8") as f:
            self.int2name = json.load(f)
        # Convert the labels back to int
        self.int2name = dict([(int(label), name) for label, name in self.int2name.items()])
        assert type(self.int2name) == dict
        log.info("int2name loaded from: %s" % file_n)

    def save_all_json(self, export_imagenet_img: bool = False) -> None:
        self._save_index_json()
        self._imagenet_save_to_file(export_imagenet_img)
        self._utk_save_to_file()
        self._vgg_save_to_file()
        self._vgg_utk_val_save_to_file()

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
        """
        Select random classes for imagenet, and create index dictionary
        :return: List of class IDs to use for imagenet
        """
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

    def _imagenet_load_images_json(self) -> None:
        # Load id2int
        file_n = os.path.join(self._imagenet_post_proc_dir, "id2int.json")
        with open(file_n, "r", encoding="utf-8") as f:
            self.imagenet_id2int = json.load(f)
        # Convert the labels back to int
        self.imagenet_id2int = dict([(cls_id, int(label)) for cls_id, label in self.imagenet_id2int.items()])
        assert type(self.imagenet_id2int) == dict
        log.info("Tiny ImageNet id2int loaded from: %s" % file_n)

        # Load id2name
        file_n = os.path.join(self._imagenet_post_proc_dir, "id2name.json")
        with open(file_n, "r", encoding="utf-8") as f:
            self._imagenet_id2name = json.load(f)
        assert type(self.int2name) == dict
        log.info("Tiny ImageNet id2name loaded from: %s" % file_n)

        # Load training data
        train_f_n = os.path.join(self._imagenet_post_proc_dir, "train.json")
        with open(train_f_n, "r", encoding="utf-8") as f:
            self.imagenet_train = json.load(f)
        assert type(self.imagenet_train) == list
        log.info("Tiny ImageNet training list loaded from: %s" % train_f_n)

        # Load validation data
        val_f_n = os.path.join(self._imagenet_post_proc_dir, "validation.json")
        with open(val_f_n, "r", encoding="utf-8") as f:
            self.imagenet_val = json.load(f)
        assert type(self.imagenet_val) == list
        log.info("Tiny ImageNet validation list loaded from: %s" % val_f_n)

    def _imagenet_save_to_file(self, export_img: bool = False) -> None:
        # Save id2int
        file_n = os.path.join(self._imagenet_post_proc_dir, "id2int.json")
        with open(file_n, "w", encoding="utf-8") as f:
            json.dump(self.imagenet_id2int, f)
        log.info("Tiny ImageNet id2int saved as: %s" % file_n)

        # Save id2name
        file_n = os.path.join(self._imagenet_post_proc_dir, "id2name.json")
        with open(file_n, "w", encoding="utf-8") as f:
            json.dump(self._imagenet_id2name, f)
        log.info("Tiny ImageNet id2name saved as: %s" % file_n)

        # Save training dataset
        train_f_n = os.path.join(self._imagenet_post_proc_dir, "train.json")
        with open(train_f_n, "w", encoding="utf-8") as f:
            json.dump(self.imagenet_train, f)
        log.info("Tiny ImageNet training list saved as: %s" % train_f_n)

        # Save validation dataset
        val_f_n = os.path.join(self._imagenet_post_proc_dir, "validation.json")
        with open(val_f_n, "w", encoding="utf-8") as f:
            json.dump(self.imagenet_val, f)
        log.info("Tiny ImageNet validation list saved as: %s" % val_f_n)

        if export_img:
            os.makedirs(os.path.join(self._imagenet_post_proc_dir, "val", "images"), exist_ok=True)
            # Copy all training files to post proc dir
            for file_n, cls_id in self.imagenet_train:
                full_file_name = os.path.join(self.imagenet_dir, "train", cls_id, "images", file_n)
                os.makedirs(os.path.join(self._imagenet_post_proc_dir, "train", cls_id, "images"), exist_ok=True)
                shutil.copy2(full_file_name,
                             os.path.join(self._imagenet_post_proc_dir, "train", cls_id, "images", file_n))
            # Copy all validation files to post proc dir
            for file_n, cls_id in self.imagenet_val:
                full_file_name = os.path.join(self.imagenet_dir, "val", "images", file_n)
                shutil.copy2(full_file_name, os.path.join(self._imagenet_post_proc_dir, "val", "images", file_n))

    def import_imagenet_images(self) -> None:
        os.makedirs(os.path.join(self.imagenet_dir), exist_ok=True)
        # Copy all training files to pre proc dir
        shutil.move(os.path.join(self._imagenet_post_proc_dir, "train"), os.path.join(self.imagenet_dir, "train"))
        # Copy all validation files to pre proc dir
        shutil.move(os.path.join(self._imagenet_post_proc_dir, "val"), os.path.join(self.imagenet_dir, "val"))

    def _utk_load_images(self) -> None:
        """
        Iterate through utk images and parse data
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

    def _utk_update_images(self) -> None:
        """
        Resize utk images
        :return: None
        """
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
        log.debug("Resized validation images. Saved at %s" % val_dir)

    def _utk_load_images_json(self) -> None:
        """
        Load UTKFace dataset from files
        :return: None
        """
        # Load training data
        train_f_n = os.path.join(self.utk_post_proc_dir, "train.json")
        with open(train_f_n, "r", encoding="utf-8") as f:
            self.utk_train = json.load(f)
        # Convert the labels back to int
        self.utk_train = [(img, int(label)) for img, label in self.utk_train]
        assert type(self.utk_train) == list
        log.info("UTKFace training list loaded from: %s" % train_f_n)

        # Load validation data
        val_f_n = os.path.join(self.utk_post_proc_dir, "validation.json")
        with open(val_f_n, "r", encoding="utf-8") as f:
            self.utk_val = json.load(f)
        # Convert the labels back to int
        self.utk_val = [(img, int(label)) for img, label in self.utk_val]
        assert type(self.utk_val) == list
        log.info("UTKFace validation list loaded from: %s" % val_f_n)

    def _utk_save_to_file(self) -> None:
        """
        Save UTKFace dataset to files
        :return: None
        """
        # Save training data
        train_f_n = os.path.join(self.utk_post_proc_dir, "train.json")
        with open(train_f_n, "w", encoding="utf-8") as f:
            json.dump(self.utk_train, f)
        log.info("UTKFace training list saved as: %s" % train_f_n)

        # Save validation data
        val_f_n = os.path.join(self.utk_post_proc_dir, "validation.json")
        with open(val_f_n, "w", encoding="utf-8") as f:
            json.dump(self.utk_val, f)
        log.info("UTKFace validation list saved as: %s" % val_f_n)

    def vgg_download_images(self) -> None:
        """
        Download images using VGG dataset text files
        :return: None
        """
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
        self.vgg_train = _images[:self._train_img_count]
        self.vgg_val = _images[self._train_img_count:self._train_img_count + val_size]
        log.debug("Length of vgg training images: %s" % len(self.vgg_train))
        log.debug("Length of vgg validating images: %s" % len(self.vgg_val))

    def _vgg_update_images(self) -> None:
        # create modified validation data directory
        val_dir = os.path.join(self.vgg_post_proc_dir, "val")
        os.makedirs(val_dir, exist_ok=True)

        # create modified training data directory
        train_dir = os.path.join(self.vgg_post_proc_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # Resize all training dataset images
        for img_n, _ in self.vgg_train:
            with Image.open(os.path.join(self.vgg_post_proc_dir, 'downloaded', img_n)) as img:
                img = img.resize((self._img_size, self._img_size))
                img.save(os.path.join(train_dir, img_n))

        # Resize all validation dataset images
        for img_n, _ in self.vgg_val:
            with Image.open(os.path.join(self.vgg_post_proc_dir, 'downloaded', img_n)) as img:
                img = img.resize((self._img_size, self._img_size))
                img.save(os.path.join(val_dir, img_n))
        log.debug("Resized images saved.")

    def _vgg_load_images_json(self) -> None:
        # Load training data
        train_f_n = os.path.join(self.vgg_post_proc_dir, "train.json")
        with open(train_f_n, "r", encoding="utf-8") as f:
            self.vgg_train = json.load(f)
        # Convert the labels back to int
        self.vgg_train = [(img, int(label)) for img, label in self.vgg_train]
        assert type(self.vgg_train) == list
        log.info("VGG training list loaded from: %s" % train_f_n)

        # Load validation data
        val_f_n = os.path.join(self.vgg_post_proc_dir, "validation.json")
        with open(val_f_n, "r", encoding="utf-8") as f:
            self.vgg_val = json.load(f)
        # Convert the labels back to int
        self.vgg_val = [(img, int(label)) for img, label in self.vgg_val]
        assert type(self.vgg_val) == list
        log.info("VGG validation list loaded from: %s" % val_f_n)

    def _vgg_save_to_file(self) -> None:
        train_f_n = os.path.join(self.vgg_post_proc_dir, "train.json")
        with open(train_f_n, "w", encoding="utf-8") as f:
            json.dump(self.vgg_train, f)
        log.info("VGG training list saved as: %s" % train_f_n)

        val_f_n = os.path.join(self.vgg_post_proc_dir, "validation.json")
        with open(val_f_n, "w", encoding="utf-8") as f:
            json.dump(self.vgg_val, f)
        log.info("VGG validation list saved as: %s" % val_f_n)

    def _load_vgg_utk_val(self) -> None:
        # append all validation data
        for file_n, label in self.utk_val:
            self.vgg_utk_val.append((os.path.join(self.utk_post_proc_dir, "val", file_n), label))
        for file_n, label in self.vgg_val:
            self.vgg_utk_val.append((os.path.join(self.vgg_post_proc_dir, "val", file_n), label))
        for file_n, label_id in self.imagenet_val:
            self.vgg_utk_val.append((file_n, label_id))

    def _vgg_utk_load_images_json(self) -> None:
        # Load VGG-UTK validation data
        val_f_n = os.path.join(self._vgg_utk_val_dir, "validation.json")
        with open(val_f_n, "r", encoding="utf-8") as f:
            self.vgg_utk_val = json.load(f)
        # Convert the labels back to int if the label is digit
        self.vgg_utk_val = [(img, int(label)) if str(label).isdigit() else (img, label) for img, label in
                            self.vgg_utk_val]
        assert type(self.vgg_utk_val) == list
        log.info("VGG-UTK validation list loaded from: %s" % val_f_n)

    def _vgg_utk_val_save_to_file(self) -> None:
        val_f_n = os.path.join(self._vgg_utk_val_dir, "validation.json")
        with open(val_f_n, "w", encoding="utf-8") as f:
            json.dump(self.vgg_utk_val, f)
        log.info("VGG-UTK validation list saved as: %s" % val_f_n)


# Just for testing
if __name__ == '__main__':
    test = DatasetUtil(base_dir="../datasets/")
    # test.save_all_json(True)

    # test.imagenet_save_to_file()
    # test.utk_save_to_file()
    # test.vgg_save_to_file()
    # test.vgg_download_images()
    # print(test.vgg_utk_val)

    print(test.vgg_utk_val)
    # print(test.imagenet_train[:5])
    # print(test.imagenet_val[:5])
    # print(test.utk_train[:5])
    # print(test.utk_val[:5])
    # print(test.vgg_train[:5])
    # print(test.vgg_val[:5])
