import io
import logging
import math
import os
import random

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
                 vgg_dataset_dir: str = "vgg_face_dataset", img_seed: int = 10, total_dataset_size: int = 550):
        random.seed(img_seed)
        self._base_dir = base_dir
        self._base_dataset = base_dataset_dir
        self._vgg_dir = vgg_dataset_dir
        self._dataset_size = total_dataset_size

    def __getitem__(self, item) -> (torch.Tensor, int):
        pass

    def __len__(self) -> int:
        return self._dataset_size

    def download_images(self, size: int = 64) -> None:
        total = 0
        tmp_dir = os.path.join(self._base_dir, self._vgg_dir)
        img_dir = os.path.join(tmp_dir, "processed")
        os.makedirs(img_dir, exist_ok=True)
        allowed_img_ext = [".jpg", ".jpeg", ".png"]
        files = os.listdir(os.path.join(tmp_dir, "files"))
        log.debug("File list: %s" % str(files))
        # Use random files
        random.shuffle(files)
        while total < 500:
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
                                with requests.get(url) as res:
                                    log.debug("Response code: %s\tSize:%s" % (str(res.status_code),
                                                                              str(len(res.content))))
                                    # If status code is normal and content size is not 0
                                    if res.status_code == 200 and len(res.content) != 0:
                                        try:
                                            with Image.open(io.BytesIO(res.content)) as img:
                                                img = img.crop((left, top, right, bot))
                                                img.save(os.path.join(img_dir, img_name))
                                                img.close()
                                                downloaded = True
                                                total += 1
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


if __name__ == '__main__':
    data = ImagenetVgg()
    data.download_images()
