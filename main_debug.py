# %% md

## Import statements

# %%

# from google.colab import drive
# drive.mount('/content/drive')

# %%

# !pwd
# !ls drive/MyDrive/CSS490/
# !mkdir ./datasets
# !cp drive/MyDrive/CSS490/modified_datasets.tar.gz ./datasets
# !mkdir ./util
# !cp -r drive/MyDrive/CSS490/util/ ./
# !tar -xf ./datasets/modified_datasets.tar.gz --directory=./datasets/
# !python --version

# %%

# !mkdir. / datasets
# !mkdir. / util
# # Clone unstable branch
# !git
# clone - b
# unstable
# https: // github.com / seo - chang / CSS490_Final_Project
# # Copy in case of pulling
# !cp - r. / CSS490_Final_Project / util /./
# !curl - OL
# https: // github.com / seo - chang / CSS490_Final_Project / releases / latest / download / modified_datasets.zip
# !unzip - qn. / modified_datasets.zip - d. / datasets
# !ls
# !python - -version

# %%

import copy
import math
import os
import random
import time
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models

from util.dataset_util import DatasetUtil
from util.dataset_vgg_utk import ValDatasetVggUtk
from util.imagenet import Imagenet
from util.imagenet_utk import ImagenetUtk
from util.imagenet_vgg import ImagenetVgg

# Use CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using %s for training/validating the model" % device)

# Reproducibility related settings
# See https://pytorch.org/docs/stable/notes/randomness.html for more info.
seed = 1029
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# %% md
## Data Preprocessing

# %%

# Parameter settings for batching
BATCH_SIZE = 128
# 64 <= image_size <= 224
IMAGE_SIZE = 64
# 0 <= WORKER_COUNT
WORKER_COUNT = 7
# 1 <= NUMBER_OF_CLASSES <= 201
NUMBER_OF_CLASSES = 10
# 0 < TRAIN_IMAGE_COUNT <= 500
TRAIN_IMAGE_COUNT = 500
# True/False
shuffle = True

BASE_DIR = "./datasets"

# Use this line if you're executing for the first time
# data_utility = DatasetUtil(base_dir="./datasets", total_class_count=10, img_size=64, train_img_count=500,
#                            vgg_download=True)
data_utility = DatasetUtil(base_dir=BASE_DIR, total_class_count=NUMBER_OF_CLASSES, img_size=IMAGE_SIZE,
                           train_img_count=TRAIN_IMAGE_COUNT,
                           load_from_json=False)
data_utility.save_all_json()

imagenet_datasets = {'train': Imagenet(du=data_utility, base_dir=BASE_DIR, image_size=IMAGE_SIZE),
                     'val': Imagenet(du=data_utility, base_dir=BASE_DIR, image_size=IMAGE_SIZE, validation=True)}
imagenet_dataloaders = {'train': DataLoader(dataset=imagenet_datasets['train'], batch_size=BATCH_SIZE,
                                            shuffle=shuffle, num_workers=WORKER_COUNT),
                        'val': DataLoader(dataset=imagenet_datasets['val'], batch_size=BATCH_SIZE,
                                          shuffle=shuffle, num_workers=WORKER_COUNT)}

# Load data for Tiny ImageNet + UTKFace
utk_datasets = {'train': ImagenetUtk(du=data_utility, base_dir=BASE_DIR, image_size=IMAGE_SIZE),
                'val': ImagenetUtk(du=data_utility, base_dir=BASE_DIR, image_size=IMAGE_SIZE, validation=True)}
utk_dataloaders = {'train': DataLoader(dataset=utk_datasets['train'], batch_size=BATCH_SIZE,
                                       shuffle=shuffle, num_workers=WORKER_COUNT),
                   'val': DataLoader(dataset=utk_datasets['val'], batch_size=BATCH_SIZE,
                                     shuffle=shuffle, num_workers=WORKER_COUNT)}

# Load data for Tiny ImageNet + VGG
vgg_datasets = {'train': ImagenetVgg(du=data_utility, base_dir=BASE_DIR, image_size=IMAGE_SIZE),
                'val': ImagenetVgg(du=data_utility, base_dir=BASE_DIR, image_size=IMAGE_SIZE, validation=True)}
vgg_dataloaders = {'train': DataLoader(dataset=vgg_datasets['train'], batch_size=BATCH_SIZE,
                                       shuffle=shuffle, num_workers=WORKER_COUNT),
                   'val': DataLoader(dataset=vgg_datasets['val'], batch_size=BATCH_SIZE,
                                     shuffle=shuffle, num_workers=WORKER_COUNT)}

utk_vgg_dataset = ValDatasetVggUtk(du=data_utility, image_size=IMAGE_SIZE)
utk_vgg_dataloader = DataLoader(dataset=utk_vgg_dataset, batch_size=BATCH_SIZE,
                                shuffle=shuffle, num_workers=WORKER_COUNT)


# %% md

## Dataset preview

# %%

def info_data(image_datasets: dict, dataloaders: dict) -> None:
    """
    Prints info. about datasets/dataloaders; uncomment if necessary
    :return: None
    """
    cls = image_datasets["train"].get_class_names()
    print('Number of classes: %s' % len(cls))
    print('Class names: %s' % cls)
    print('Length of training dataset: %s' % len(image_datasets['train']))
    print('Length of validation dataset: %s' % len(image_datasets['val']))
    print('Batch size: %s' % BATCH_SIZE)
    print('Number of batches in the training dataloader: %s' % len(dataloaders['train']))
    print('Number of batches in the training dataloader: %s' % len(dataloaders['val']))
    print('Device: %s\n' % device)


print("Imagenet dataset statistics")
info_data(imagenet_datasets, imagenet_dataloaders)
print("UTKFace dataset statistics")
info_data(utk_datasets, utk_dataloaders)
print("VGG Face dataset statistics")
info_data(vgg_datasets, vgg_dataloaders)


# %%

def image_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    fig1 = plt.figure(figsize=(10, 10), dpi=100)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.imshow(inp)
    if title is not None:
        ax1.set_title(title)


def get_preview(dataloaders: dict, title: str):
    """
    Preview 64 images in training dataset
    :param dataloaders: Dataloader dictionary to use to print images
    :param title: Title of the preview
    :return: None
    """
    # Get the first batch of training data
    inputs, classes = list(dataloaders['train'])[0]

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs[:64], normalize=True)

    image_show(out, title=title)


# %%

get_preview(dataloaders=utk_dataloaders, title="UTKFace Preview")

# %%

get_preview(dataloaders=vgg_dataloaders, title="VGGFace Preview")

# %% md

## Model Training

# %%

# load models; this will save time

# utk_model_ft = models.resnet50()
# vgg_model_ft = models.resnet50()

# utk_model_ft.load_state_dict(torch.load('utk_model_weights.pth'))
# vgg_model_ft.load_state_dict(torch.load('vgg_model_weights.pth'))
# utk_model_ft.eval()
# vgg_model_ft.eval()

# %%


# Parameters
EPOCH = 25
LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 0.0001
LR_DECAY_STEP_SIZE = 7
LR_DECAY_FACTOR = 0.1


def train_model(model_ft: Any, dataloaders, image_datasets, verbose=False):
    def _train_model(model: Any, criterion, optimizer, scheduler, dl, img_datasets):
        since = time.time()
        stats = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(EPOCH):
            stat = []
            if verbose:
                print('Epoch {}/{}'.format(epoch, EPOCH - 1))
                print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0  # Positive
                running_incorrects = 0  # Negative

                # Iterate over data.
                for imgs, labels in dl[phase]:
                    # print('Iterating ', labels, '...')
                    torch.cuda.empty_cache()  # clean up cache
                    # print(torch.cuda.memory_summary(device=device, abbreviated=False))
                    imgs = imgs.float().to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * imgs.size(0)
                    running_corrects += torch.eq(preds, labels.data).sum()
                    running_incorrects += torch.not_equal(preds, labels.data).sum()

                    # print(running_loss, running_corrects)

                if phase == 'train':
                    scheduler.step()
                dataset_size = len(img_datasets[phase])
                """
                running_corrects.double() = True (True positive + True negative) 
                running_incorrects.double() = False (False positive + False negative)
                """

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corrects.double() / dataset_size
                epoch_tn = running_incorrects.double() / dataset_size

                if verbose:
                    print("Running corrects: %s" % running_corrects.item())
                    print("Running incorrects: %s" % running_incorrects.item())
                    print("Dataset size: %s" % dataset_size)
                    print('{} Loss: {:.4f} Acc: {:.2f}%\n'.format(
                        phase, epoch_loss, epoch_acc * 100))
                stat.append(epoch_loss)
                stat.append(epoch_acc.item())
                # print('this is stat: ' + str(stat))
                # nb_classes = 2 #??
                # confusion_matrix = torch.zeros(nb_classes, nb_classes)
                # deep copy the model
                if phase == 'val':
                    # print(stats)
                    # inputs = inputs.to(device)
                    # classes = classes.to(device)
                    # outputs = model_ft
                    stats.append(stat)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:2f}'.format(best_acc * 100))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, stats

    num_ftrs = model_ft.fc.in_features

    # TODO: Here the size of each output sample is set to 2 it is the number of classes.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(image_datasets['train'].get_class_names()))

    model_ft = model_ft.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, weight_decay=SGD_WEIGHT_DECAY,
                             momentum=SGD_MOMENTUM)

    # Decay LR by a factor of 0.1 every 7 epochs
    # TODO: Perhaps we can consider ReduceLROnPlateau instead
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=LR_DECAY_STEP_SIZE, gamma=LR_DECAY_FACTOR)

    return _train_model(model_ft, loss_fn, optimizer_ft, exp_lr_scheduler,
                        dl=dataloaders, img_datasets=image_datasets)


# model_arch = models.resnet50()

# %%

imgnet_model_ft, imgnet_stats = train_model(models.resnet50(), dataloaders=imagenet_dataloaders,
                                            image_datasets=imagenet_datasets,
                                            verbose=True)

# %%

utk_model_ft, utk_stats = train_model(models.resnet50(), dataloaders=utk_dataloaders, image_datasets=utk_datasets,
                                      verbose=True)

# %%

# To see if we can get the same accuracy
# utk_model_ft2, _ = train_model(models.resnet50(), dataloaders=utk_dataloaders, image_datasets=utk_datasets,
#                                verbose=True)

# %%

vgg_model_ft, vgg_stats = train_model(models.resnet50(), dataloaders=vgg_dataloaders, image_datasets=vgg_datasets,
                                      verbose=True)


# %%

# Save stats
def save_stats(s: List[List], file_n: str) -> None:
    with open(file_n, "w", encoding="utf-8") as f:
        for elem in s:
            f.write("%s,%s,%s,%s\n" % (elem[0], elem[1], elem[2], elem[3]))


save_stats(imgnet_stats, "imgnet_stats.csv")
save_stats(utk_stats, "utk_stats.csv")
save_stats(vgg_stats, "vgg_stats.csv")

# %%

# Save models
torch.save(imgnet_model_ft.state_dict(), 'imgnet_model_weights.pth')
torch.save(utk_model_ft.state_dict(), 'utk_model_weights.pth')
torch.save(vgg_model_ft.state_dict(), 'vgg_model_weights.pth')


# %% md

## Model Visualization

# %%

def visualize_model(model, datasets: dict, dataloaders: dict, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig2 = plt.figure(figsize=(10, 15), dpi=100)
    row = math.ceil(num_images / 2)
    col = 2
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloaders['val']):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for j in range(imgs.size()[0]):
                images_so_far += 1

                pred_str = datasets["train"].get_class_name(preds[j].item()).split(",")[0]
                gt_str = datasets["train"].get_class_name(labels.cpu().numpy()[images_so_far - 1]).split(",")[0]

                output_img = imgs.int().cpu().data[j].numpy().transpose((1, 2, 0))
                ax2 = fig2.add_subplot(row, col, images_so_far)
                ax2.imshow(output_img)
                ax2.axis('off')
                ax2.set_title(('Pred:%s GT:%s' % (pred_str, gt_str)))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# %%

# Visualize utk models
visualize_model(utk_model_ft, utk_datasets, utk_dataloaders)

# %%

# Visualize vgg models
visualize_model(vgg_model_ft, vgg_datasets, vgg_dataloaders)


# make predictions for utk_vgg dataset
def model_prediction(model):
    was_training = model.training
    # correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results = np.zeros((10, 10), dtype=int)
    model.eval()
    run_cnt = 50
    for x in range(run_cnt):
        print("Current run: %s" % x)
        # Disable gradient calculation
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(utk_vgg_dataloader):
                imgs = imgs.float().to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

                for i in range(preds.size()[0]):
                    results[utk_vgg_dataset[i][1]][preds[i].item()] += 1
                    total += 1
                model.train(mode=was_training)

        # print(results)

        for i in range(len(results)):  # get true positive (all correctly classified faces)
            tp += results[i][i]

        for i in range(len(results) - 1):  # true negative (all things that that are correctly classified as non-faces)
            for j in range(len(results) - 1):
                tn += results[j][i]

        for i in range(len(results) - 1):  # false positive (all other things incorrectly classified as face)
            fp += results[i][9]

        for i in range(len(results) - 1):  # false negative (all faces incorrectly classified as non-face)
            fn += results[9][i]

    total = total // run_cnt
    tp = tp // run_cnt
    tn = tn // run_cnt
    fp = fp // run_cnt
    fn = fn // run_cnt
    print(tp, tn, fp, fn)

    results = results / run_cnt

    # for i in range(len(results)): # get false positive; iterate through each column
    #     for j in range(len(results[i])):
    #         if j != i:
    #             false_positive_img_count += results[j][i]

    def show2():
        cf_matrix = np.array(([tp, fp], [fn, tn]), dtype=int)
        # print(cf_matrix)

        group_names = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
        group_counts = ['{0:0.0f}'.format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in
                             cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                  zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.set(font_scale=2)
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)

    def show10():
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel('Predicted Values', fontsize=20.0)
        ax.set_xlabel('Actual Values', fontsize=20.0)
        ax.set_aspect('equal')
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False,
                        labelsize=20)

        for i in range(len(results)):
            for j in range(len(results[i])):
                text = ax.text(j, i, results[j, i],
                               ha="center", va="center", color="black", fontsize=12)
        ax.imshow(results, interpolation='nearest', cmap='Blues')

    show2()
    show10()


# %%

model_prediction(utk_model_ft)

# %%
model_prediction(vgg_model_ft)


# %%

def correctness(model):  # get correctness for faces
    was_training = model.training
    # correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results = np.zeros((10, 10), dtype=int)
    model.eval()
    # Disable gradient calculation
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(utk_vgg_dataloader):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for i in range(preds.size()[0]):
                results[utk_vgg_dataset[i][1]][preds[i].item()] += 1
                total += 1
            model.train(mode=was_training)
    print(results)

    for i in range(len(results)):  # get true positive (all correctly classified faces)
        tp += results[i][i]

    for i in range(len(results) - 1):  # true negative (all things that that are correctly classified as non-faces)
        for j in range(len(results) - 1):
            tn += results[j][i]

    for i in range(len(results) - 1):  # false positive (all other things incorrectly classified as face)
        fp += results[i][9]

    for i in range(len(results) - 1):  # false negative (all faces incorrectly classified as non-face)
        fn += results[9][i]

    # for i in range(len(results)): # get false positive; iterate through each column
    #     for j in range(len(results[i])):
    #         if j != i:
    #             false_positive_img_count += results[j][i]

    print('tp: ', tp)
    print('tn: ', tn)
    print('fp: ', fp)
    print('fn: ', fn)

    return tp, tn, fp, fn


# %% md
# ROC Curve

# %%

def _plot_roc_curve(tpr, fpr):
    plt.figure(1, figsize=(6, 5))
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.plot(fpr, tpr, color='red')
    plt.legend(loc="lower right")
    plt.show()
    # sns.set(font_scale= 1)


tp, tn, fp, fn = correctness(vgg_model_ft)
# calculate true positive rate
tpr = tp / (tp + fn)
print('tpr: ', tpr)
# Calculate false positive rate
fpr = fp / (fp + tn)
print('fpr: ', fpr)
plot = _plot_roc_curve(tpr, fpr)
