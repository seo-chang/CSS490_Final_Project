import PIL.Image
import numpy as np
import torch.utils.data


class TempData(torch.utils.data.Dataset):

    def __init__(self, file_n: str):
        image = PIL.Image.open(file_n).convert("RGB")
        arr = np.transpose(np.array(image), (2, 0, 1))
        # print(arr.shape)
        self.example = torch.from_numpy(arr).float()

    def __getitem__(self, item):
        return self.example

    def __len__(self):
        return 1
