"""
Fine-tune from ResNet18 ImageNet weights two ways: 
(1) fine-tune all layers (2) just the fully-connected layer
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
import copy
from utils import imshow, load_data


def train_val_model(model, criterion, optimizer, scheduler, num_epochs=10):

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
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

            # Iterate over mini-batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # zero out gradients

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

            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

    print('Best validation Accuracy: {:4f}'.format(best_accuracy))
    model.load_state_dict(best_model_weights)  # retain best weights
    return model


def test_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
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


def fine_tune_model(mode):
    criterion = nn.CrossEntropyLoss()
    model = models.resnet18(pretrained=True)  # pretrained=True will download its weights

    num_in_features_last = model.fc.in_features
    
    if mode == 'all_layers':  # fine-tune all layers
        model.fc = nn.Linear(num_in_features_last, 2)  # make last layer a binary classifier
        # Note, parameters in all layer are being optimized
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif mode == 'only_fc_layer':
        for param in model.parameters():
            param.requires_grad = False
            
        # Newly constructed module has requires_grad=True by default
        model.fc = nn.Linear(num_in_features_last, 2)
        
        # Note, only parameters of final layer are being optimized
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    else:
        print('Unknown training mode')
        exit(2)
    model = model.to(device)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_val_model(model, criterion, optimizer, exp_lr_scheduler)
    return model

plt.ion()   # interactive mode

dataloaders, dataset_sizes, class_names = load_data('superbeings')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get a batch of training data and show it
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

model = fine_tune_model('all_layers')
test_model(model)
model = fine_tune_model('only_fc_layer')  # train just the fully-connected layer
test_model(model)

plt.ioff()
plt.show()
