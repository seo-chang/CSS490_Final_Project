import json
import math
import os
import random

import PIL.Image


class UTKFaceUtil:
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
    test = UTKFaceUtil()
