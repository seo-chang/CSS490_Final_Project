import os.path

import PIL
import numpy as np
import torch.utils.data
import torchvision.transforms as vision

from util.dataset_util import DatasetUtil


class ValDatasetVggUtk(torch.utils.data.Dataset):
    def __init__(self, du: DatasetUtil, image_size: int = 224):
        self._du = du
        self._images = []  # [("n01443537_0.jpeg", 7335), ...]
        self._image_size = image_size
        self._images.extend(du.vgg_utk_val)

    def __len__(self) -> int:
        """
        Get length of the dataset
        :return: Total image count
        """
        return len(self._images)

    def __getitem__(self, idx) -> (torch.Tensor, int):
        full_file_name = self._images[idx][0]

        cls_int = self._images[idx][1]
        # if current data is imagenet validation data
        if type(cls_int) == str:
            cls_int = self._du.imagenet_id2int[cls_int]
            full_file_name = os.path.join(self._du.imagenet_dir, "val", "images", full_file_name)

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
    test = ValDatasetVggUtk(data_util)
    print(test[10])
