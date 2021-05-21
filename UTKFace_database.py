import os

import PIL.Image
import numpy as np
import torch.utils.data
import random


class TempData(torch.utils.data.Dataset):
    def __init__(self, base_dir: str = './dataset/', file_n: str = 'UTKFace', img_seed: int = 10):
        """
        Initialize the dataset
        :param base_dir: Dataset directory
        :param file_n: File name
        :param img_seed:
        """
        random.seed(img_seed)
        # image = PIL.Image.open(file_n).convert("RGB")
        # arr = np.transpose(np.array(image), (2, 0, 1))
        # print(arr.shape)
        # self.example = torch.from_numpy(arr).float()
        self._images = []

        self._training = {"white": [],  # 0
                          "black": [],  # 1
                          "asian": [],  # 2
                          "indian": [],  # 3
                          "male": [],  # 0
                          "female": []  # 1
                          }
        self._total_training = []
        self._white = []
        self._black = []
        self._asian = []
        self._indian = []
        self._male = []
        self._female = []

        self._src = os.path.join(base_dir, file_n)
        # [race] is an integer from 0 to 4, denoting White, Black, Asian, and Indian
        # [gender] is either 0 (male) or 1 (female)
        self._d = {"white": [],  # 0
                   "black": [],  # 1
                   "asian": [],  # 2
                   "indian": [],  # 3
                   "male": [],  # 0
                   "female": []  # 1
                   }

        self._load_images(self._src)

    def _total_training(self):
        return self._total_training

    def __getitem__(self, item) -> (torch.Tensor, int):
        """"

        """
        return self.example

    def __len__(self):
        """
        Get length of the dataset
        :return: Total image count
        """
        return len(self._images)

    def _shuffle(self, feature: str, image_per_class):
        """
        :param feature:
        :return:
        We decided to use static int (500). 500 = images per class, 50 = number of validations set
        """
        num = []
        count = 0
        while count < (round(image_per_class / len(self._d))):
            rand_num = random.randint(0, len(self._d[feature]))
            if rand_num not in num and self._d[feature][rand_num] not in self._total_training:
                num.append(rand_num)
                self._total_training.append(self._d[feature][rand_num])
                self._training[feature].append(self._d[feature][rand_num])
                count += 1

    def _load_images(self, _dir):
        """
        :param _dir:
        :return:
        We decided to erase other(race) because data was empty for other(race)
        """
        file_names = os.listdir(_dir)
        for file_name in file_names:
            sep = file_name.split('_')
            # print(sep)
            # print(type(sep[1]))
            if sep[1] == '0':
                self._d['male'].append(file_name)
            if sep[1] == '1':
                self._d['female'].append(file_name)
            if sep[2] == '0':
                self._d['white'].append(file_name)
            if sep[2] == '1':
                self._d['black'].append(file_name)
            if sep[2] == '3':
                self._d['asian'].append(file_name)
            if sep[2] == '4':
                self._d['indian'].append(file_name)
            self._images.append(file_name)

        for feature in self._d:
            self._shuffle(feature, 500)
        print(len(self._total_training))
        print(len(set(self._total_training)))


class validataion(torch.utils.data.Dataset):
    def __init__(self, base_dir: str = './dataset/', file_n: str = 'UTKFace', img_seed: int = 10,
                 list_of_testing: list = []):
        """
        Initialize the dataset
        :param base_dir: Dataset directory
        :param file_n: File name
        :param img_seed:
        :param list_of_testing: list contains testing data (file name)
        """
        random.seed(img_seed)
        # image = PIL.Image.open(file_n).convert("RGB")
        # arr = np.transpose(np.array(image), (2, 0, 1))
        # print(arr.shape)
        # self.example = torch.from_numpy(arr).float()
        self._images = []

        self._validation = {"white": [],  # 0
                            "black": [],  # 1
                            "asian": [],  # 2
                            "indian": [],  # 3
                            "male": [],  # 0
                            "female": []  # 1
                            }

        self._total_training = list_of_testing
        self._white = []
        self._black = []
        self._asian = []
        self._indian = []
        self._male = []
        self._female = []

        self._src = os.path.join(base_dir, file_n)
        # [race] is an integer from 0 to 4, denoting White, Black, Asian, and Indian
        # [gender] is either 0 (male) or 1 (female)
        self._d = {"white": [],  # 0
                   "black": [],  # 1
                   "asian": [],  # 2
                   "indian": [],  # 3
                   "male": [],  # 0
                   "female": []  # 1
                   }

        self._load_images(self._src)

    def __getitem__(self, item) -> (torch.Tensor, int):
        """"

        """
        return self.example

    def __len__(self):
        """
        Get length of the dataset
        :return: Total image count
        """
        return len(self._images)

    def _shuffle(self, feature: str, image_per_class):
        """
        :param feature:
        :return:
        We decided to use static int (500). 500 = images per class, 50 = number of validations set
        """
        num = []
        count = 0
        while count < (round(image_per_class / len(self._d))):
            rand_num = random.randint(0, len(self._d[feature]))
            if rand_num not in num and self._d[feature][rand_num] not in self._total_training:
                num.append(rand_num)
                self._total_training.append(self._d[feature][rand_num])
                self._validation[feature].append(self._d[feature][rand_num])
                count += 1

    def _load_images(self, _dir):
        """
        :param _dir:
        :return:
        We decided to erase other(race) because data was empty for other(race)
        """
        file_names = os.listdir(_dir)
        for file_name in file_names:
            sep = file_name.split('_')
            # print(sep)
            # print(type(sep[1]))
            if sep[1] == '0':
                self._d['male'].append(file_name)
            if sep[1] == '1':
                self._d['female'].append(file_name)
            if sep[2] == '0':
                self._d['white'].append(file_name)
            if sep[2] == '1':
                self._d['black'].append(file_name)
            if sep[2] == '3':
                self._d['asian'].append(file_name)
            if sep[2] == '4':
                self._d['indian'].append(file_name)
            self._images.append(file_name)

        for feature in self._d:
            self._shuffle(feature, 56)
        print(len(self._total_training))
        print(len(set(self._total_training)))
        print(len(self._validation))


# Just for testing
if __name__ == '__main__':
    test = TempData()
    print(test._total_training)
    _validataion = validataion( './dataset/', 'UTKFace',  10, test._total_training)
    print(len(_validataion._total_training))
    print(len(set(_validataion._total_training)))
    # test.__len__()
    # test._shuffle()
    # test.__listDir__()
    # print(len(test))
    # print(test[0][0].shape)
    # print(test[0][0].type())
