import io
import logging
import math
import os
import random
import json
import torchvision.transforms as vision

import PIL.Image
import numpy as np
import requests
import torch.utils.data
from PIL import Image, UnidentifiedImageError
from requests import HTTPError, Timeout, TooManyRedirects

# Setup logger

log = logging.getLogger("imagenet_vgg")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", '%m/%d/%Y %I:%M:%S %p')
stream_h = logging.StreamHandler()
stream_h.setLevel(logging.DEBUG)
stream_h.setFormatter(formatter)
log.addHandler(stream_h)


class ImagenetVgg(torch.utils.data.Dataset):

    def __init__(self, base_dir: str = "../datasets", base_dataset_dir: str = "tiny-imagenet-200",
                 vgg_dataset_dir: str = "vgg_face_dataset", img_seed: int = 11, image_size=64,
                 total_dataset_size: int = 550, validation: bool= False):
        random.seed(img_seed)
        self._base_dir = base_dir
        self._base_dataset = base_dataset_dir
        self._vgg_dir = vgg_dataset_dir
        self._vgg_dir_modified_datasets = os.path.join(self._base_dir, "modified_datasets", 'vgg')
        self._images = []
        self._image_size = image_size
        self._training =[]
        self._validating = []
        self._vgg_validation = validation

        self._dataset_size = total_dataset_size

    def __getitem__(self, idx) -> (torch.Tensor, int):
        # if item name belongs in imagenet
        # if item in self._images:

        if self._vgg_validation:
            full_file_name = os.path.join(self._vgg_dir_modified_datasets, "val", self._validating[idx])
        else:
            full_file_name = os.path.join(self._vgg_dir_modified_datasets, "train", self._training[idx])

        image = PIL.Image.open(full_file_name).convert("RGB")
        crop = vision.CenterCrop(self._image_size)
        arr = np.transpose(np.array(crop(image)), (2, 0, 1))
        return torch.from_numpy(arr).float()


    def __len__(self) -> int:
        return self._dataset_size

    def download_images(self, size: int = 64) -> None:
        total = 0
        save_dir = os.path.join(self._base_dir, "modified_datasets", "vgg", "downloaded")
        os.makedirs(save_dir, exist_ok=True)
        tmp_dir = os.path.join(self._base_dir, self._vgg_dir)
        os.makedirs(os.path.join(tmp_dir, "processed"), exist_ok=True)
        allowed_img_ext = [".jpg", ".jpeg", ".png"]
        files = os.listdir(os.path.join(tmp_dir, "files"))
        log.debug("File list: %s" % str(files))
        # Use random files
        random.shuffle(files)
        while total < 700:
            file_n = files[total]
            if file_n.endswith(".txt"):
                downloaded = False
                person_n = file_n[:-4]
                # log.debug("Current file: %s" % file_n)
                with open(os.path.join(tmp_dir, "files", file_n), "r", encoding="utf-8") as f:
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
                                size <= right - left and size <= bot - top):
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
                                                    total += 1
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

    def save_to_file(self) -> None:
        for name in os.listdir(os.path.join(self._vgg_dir_modified_datasets, 'downloaded')):
            self._images.append(name)
        random.shuffle(self._images)
        self._training = self._images[:500]
        self._validating = self._images[500:550]
        print(len(self._training))
        print(len(self._validating))

    def _load_vgg_images(self, validation: bool) -> None:
        """
        Divide into training and validating datasets and save it as json
        :return: None
        """
        # create modified validation data directory
        val_dir = os.path.join(self._vgg_dir_modified_datasets, "val")
        os.makedirs(val_dir, exist_ok=True)

        # create modified training data directory
        train_dir = os.path.join(self._vgg_dir_modified_datasets, "train")
        os.makedirs(train_dir, exist_ok=True)

        if validation:
            val_f_n = os.path.join(self._vgg_dir_modified_datasets, "validation.json")
            with open(val_f_n, "w", encoding="utf-8") as f:
                json.dump(self._validating, f)
            print("Validation list saved as: %s\n" % val_f_n)

            # Resize all validation dataset images
            for img_n in self._validating:
                with PIL.Image.open(os.path.join(self._vgg_dir_modified_datasets, 'downloaded', img_n)) as img:
                    img = img.resize((self._image_size, self._image_size))
                    img.save(os.path.join(val_dir, img_n))
            print("Resized validation images. Saved at %s\n" % val_dir)

        else:
            train_f_n = os.path.join(self._vgg_dir_modified_datasets, "train.json")
            with open(train_f_n, "w", encoding="utf-8") as f:
                json.dump(self._training, f)
            print("Training list saved as: %s" % train_f_n)

            # Resize all validation dataset images
            for img_n in self._training:
                with PIL.Image.open(os.path.join(self._vgg_dir_modified_datasets, 'downloaded', img_n)) as img:
                    img = img.resize((self._image_size, self._image_size))
                    img.save(os.path.join(train_dir, img_n))
            print("Resized validation images. Saved at %s\n" % val_dir)



if __name__ == '__main__':
    data = ImagenetVgg()
    data.save_to_file()
    data._load_vgg_images(True)
    data._load_vgg_images(False)
    print(data[0])
