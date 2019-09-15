"""
Derived from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
import time
import copy
from utils import imshow, load_data
from pudb import set_trace


def train_val(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # forward pass
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
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
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def fine_tune(mode):
    criterion = nn.CrossEntropyLoss()
    model = models.resnet18(pretrained=True)  # pretrained=True will download its weights

    num_in_features_last = model.fc.in_features
    if mode == 'learn_all_layers':
        model.fc = nn.Linear(num_in_features_last, 2)  # make last layer a binary classifier
        # Note, parameters in all layer are being optimized
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif mode == 'learn_only_fc_layer':
        for param in model.parameters():
            param.requires_grad = False
            model.fc = nn.Linear(num_in_features_last, 2)  # make last layer a binary classifier
        # Note, only parameters of final layer are being optimized unlike before.
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    else:
        print('Unknown training mode')
        exit(1)
    print(model)
    print([[param[0], param[1].shape] for param in model.named_parameters()])
    model = model.to(device)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_val(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
    return model

plt.ion()   # interactive mode

# dataloaders, dataset_sizes, class_names = load_data('hymenoptera_data')
dataloaders, dataset_sizes, class_names = load_data('superbeings')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get a batch of training data and show it
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

model = fine_tune('learn_all_layers')
visualize_model(model)
model = fine_tune('learn_only_fc_layer')
visualize_model(model)

plt.ioff()
plt.show()
