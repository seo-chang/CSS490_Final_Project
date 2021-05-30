import PIL
import numpy as np
import torchvision.transforms as vision

import os

import torch.utils.data
from util.dataset_util import DatasetUtil


class val_dataset_vgg_utk(torch.utils.data.Dataset):
    def __init__(self, du: DatasetUtil):
        self._du = du
        self._images = []  # [("n01443537_0.jpeg", 7335), ...]
        print(du.vgg_utk_val)
        self._images.extend(du.vgg_utk_val)  #  TODO double check if it's actually grabbing from util.dataset_util.vgg_utk_val

    def __getitem__(self, idx) -> (torch.Tensor, int):
        img_name = self._images[idx][0]

        cls_int = self._images[idx][1]
        full_file_name = os.path.join(self._du.vgg_utk_val_post_proc_dir, img_name)

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



if __name__ == '__main__':
    data_util = DatasetUtil(base_dir="../datasets/")
    test = val_dataset_vgg_utk(data_util)
    print(test[0])
    test = val_dataset_vgg_utk(data_util)

