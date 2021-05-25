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
import time
import os
import copy

###############################################################
# data loading / preprocessing block

"""parameters for datasets"""
batch_size = 120
shuffle = True
image_size = 64
"""parameters for models"""
num_epochs = 15  # default size is 15

# Load data for UTK
image_datasets = {'train': ImagenetUtk(base_dir="./datasets", image_size=image_size),
                  'val': ImagenetUtk(base_dir="./datasets", validation=True, image_size=image_size)}
dataloaders = {'train': DataLoader(dataset=image_datasets['train'], batch_size=batch_size, shuffle=shuffle),
                  'val': DataLoader(dataset=image_datasets['val'], batch_size=batch_size, shuffle=shuffle)}

# this list will contain class names of items[0] to [8] in string
class_names = []
# get class names
for class_int in range(0, 10):
    class_names.append(image_datasets['train'].get_class_name(class_int))

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def info_data():
    """info. about datasets/dataloaders; uncomment if necessary"""
    print('number of classes:', len(class_names))
    print('class names:', class_names)
    print('length of training dataset:', len(image_datasets['train']))
    print('length of validation dataset:', len(image_datasets['val']))
    print('batch size:', batch_size)
    print('number of batches in the training dataloader:', len(iter(dataloaders['train'])))
    print('number of batches in the training dataloader:', len(iter(dataloaders['val'])))
    print('device:', device)


info_data()  # print out dataset info, comment out if desired


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = (inp.numpy().transpose((1, 2, 0)))
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
            progress = 0.0
            for inputs, labels in dataloaders[phase]:
                # print('Iterating ', labels, '...')
                torch.cuda.empty_cache() # clean up cache
                #print(torch.cuda.memory_summary(device=device, abbreviated=False))
                inputs = inputs.float().to(device)
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
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(('predicted:' + class_names[preds[j]] +
                              '\n answer:' + class_names[labels.cpu().numpy()[images_so_far-1]]))
                imshow(inputs.int().cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet50()

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print(model_ft)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)

visualize_model(model_ft)