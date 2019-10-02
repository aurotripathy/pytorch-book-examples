"""
Fine-tune from ResNet18 ImageNet weights:
(1) learn the fully-connected layer from scratch
(2) optionally fine-tune the rest of the layers
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

        for phase in ['train', 'val']:  # each epoch does train and validate
            if phase == 'train':
                model.train()  # Set mode
            else:
                model.eval()   # Set mode

            running_losses = 0.0
            running_corrects = 0

            # Iterate over mini-batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # zero out gradients

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # forward pass
                    _, predictions = torch.max(outputs, 1)  # predictions == argmax
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_losses += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_losses / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

            print('\t{} loss: {:.4f}, {} accuracy: {:.4f}'.format(
                phase, epoch_loss, phase, epoch_accuracy))

            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())

            if phase == 'train':
                scheduler.step()

    print('Best validation accuracy: {:4f}'.format(best_accuracy))
    model.load_state_dict(best_model_weights)  # retain best weights
    return model


def test_model(model):

    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)  # predictions == argmax
            running_corrects += torch.sum(predictions == labels.data)
        acc = running_corrects.double() / dataset_sizes['test']
    print('Test accuracy: {:.4f}'.format(acc))


def configure_model(mode):

    criterion = nn.CrossEntropyLoss()
    model = models.resnet18(pretrained=True)  # pretrained=True will download its weights

    num_in_features_last = model.fc.in_features
    
    if mode == 'fine_tune_all_layers':
        model.fc = nn.Linear(num_in_features_last,nb_classes)  # make last layer a binary classifier
        # Note, parameters in all layer are being optimized
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif mode == 'fine_tune_only_fc_layer':
        for param in model.parameters():
            param.requires_grad = False
            
        # Newly constructed module has requires_grad=True by default
        model.fc = nn.Linear(num_in_features_last, nb_classes)
        
        # Note, only parameters of final layer are being optimized
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    elif mode == 'learn_from_scratch':
        model = models.resnet18(pretrained=False)  # start with random weights in all layers
        model.fc = nn.Linear(num_in_features_last, nb_classes)  # make last layer a binary classifier
        optimizer  = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.to(device)

    # Decay learning rate by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_val_model(model, criterion, optimizer, exp_lr_scheduler)
    return model

plt.ion()   # interactive mode

dataloaders, dataset_sizes, class_names = load_data('superbeings')
print('Train size {}, Val size {}, Test size {}'.format(dataset_sizes['train'],
                                                        dataset_sizes['val'],
                                                        dataset_sizes['test']))
print('Class names:{}'.format(class_names))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_classes = len(class_names)
# Get a batch of training data and show it
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

for mode in ['learn_from_scratch',
             'fine_tune_all_layers',
             'fine_tune_only_fc_layer']:
    print('\nMode: {}'.format(mode))
    model = configure_model(mode)
    test_model(model)

plt.ioff()
plt.show()
