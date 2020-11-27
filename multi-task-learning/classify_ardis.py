# ARDIS classification
# Dataset at https://ardisdataset.github.io/ARDIS/

from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from utils import normalize, display_sample_images
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.utils import shuffle
from pudb import set_trace


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, writer):
    """ This is a very classic train loop per epoch consisting of
    the forward pass, the loss calculation, the backward pass and the model update"""
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        running_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('training loss', loss / args.log_interval, epoch * len(train_loader) + batch_idx)
            print(f'Train Epoch: {epoch} ' 
                    f'[{batch_idx * len(data)}/{len(train_loader.dataset)}'
                    f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            running_loss = 0

            
def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    writer.add_scalar('Accuracy', 100. * correct / len(test_loader.dataset), epoch * len(test_loader))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='Example of how a multi scale network benifits scale invarience.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True},)


    x_train = np.loadtxt('ardis/data/ARDIS_train_2828.csv', dtype='float32') / 255  # scale
    y_train = np.loadtxt('ardis/data/ARDIS_train_labels.csv', dtype='int')
    x_train, y_train = shuffle(x_train, y_train)

    x_test = np.loadtxt('ardis/data/ARDIS_test_2828.csv', dtype='float32') / 255  # scale
    y_test = np.loadtxt('ardis/data/ARDIS_test_labels.csv', dtype='int')
    
    # Reshape to be [samples][channels][width][height]
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_train = normalize(x_train)
    y_train = np.argmax(y_train, axis=1)  # convert one-hot to index

    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    x_test = normalize(x_test)
    y_test = np.argmax(y_test, axis=1)

    print('Train length:', len(x_train), 'Train shape:', x_train.shape)
    print('Test  length:', len(x_test), 'Test shape:', x_test.shape)

    x_train = torch.from_numpy(x_train)  # to tensor
    y_train = torch.from_numpy(y_train)
    dataset_train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset_train, **kwargs)
    
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    dataset_test = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, **kwargs)

    # display_sample_images(train_loader)
    # display_sample_images(test_loader)

    model = Net().to(device)

    writer = SummaryWriter('runs/' + 'ardis')
    torch.onnx.export(model, torch.randn(1, 1, 28, 28).to(device),
                      'ardis' + '-model.onnx', output_names=['digits-class'])
    print(model)
    print('Paramter count in Model:', sum([torch.numel(p) for p in model.parameters()]))
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(model, device, test_loader, epoch, writer)
        scheduler.step()

if __name__ == '__main__':
    main()
