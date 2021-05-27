import torchvision
from torchvision import transforms, models
from util.imagenet_utk import ImagenetUtk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
import time
import os
import copy

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
###############################################################
# data loading / preprocessing block

batch_size = 18
shuffle = True

# Load data for UTK
image_datasets = {'train': ImagenetUtk(base_dir="./datasets"),
                  'val': ImagenetUtk(base_dir="./datasets", validation=True)}
dataloaders = {'train': DataLoader(dataset=image_datasets['train'], batch_size=batch_size, shuffle=shuffle),
                  'val': DataLoader(dataset=image_datasets['val'], batch_size=batch_size, shuffle=shuffle)}

# print(utk_dataset.get_class_name(utk_dataset[0][1]))
# print('the dataset size is ', len(utk_dataset))

# this classes list will contain class names of items[0] to [8] in string
class_names = []
# get class names
for class_int in range(0, 9):
    class_names.append(image_datasets['train'].get_class_name(class_int))
class_names.append("face")

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# print(class_names)
# print(len(class_names))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
############################

# print(len(utk_dataset))
# print(type(utk_dataset[0][0]))
# dl_list = iter(utk_dataloader)
# print(len(dl_list))
# i = 0
#
# inputs, classes = next(dl_list)
# print('inputs dimensions: ', inputs.ndim)
# print(inputs.shape)
# print('len input[0]: ', len(inputs[0]))
# print('type: ', type(inputs[0]))
# print(inputs[0].shape)

##############################

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = (inp.numpy().transpose((1, 2, 0))).astype('int32')
    print(len(inp))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(60)


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

################
# Training the model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                print('Iterating ', labels, '...')
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                print(inputs.cpu().data[j])
                print(type(inputs.cpu().data[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

model_ft = models.resnet50()

num_ftrs = model_ft.fc.in_features
# TODO: Here the size of each output sample is set to 2 it is the number of classes.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# print(model_ft)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=0)

visualize_model(model_ft)