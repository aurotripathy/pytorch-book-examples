""" How to detect objects at multiple scales """

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset
from utils import UnNormalize, display_sample_images
from tensorboardX import SummaryWriter


class NetOrig(nn.Module):
    """ Copied from https://github.com/pytorch/examples/blob/master/mnist/main.py """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, dilation=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
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


class NetConcatAllScales(nn.Module):
    """ This class concatenates two parallel nets. """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, dilation=2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc_scale_1 = nn.Linear(32 * 54 * 54, 128)
        self.fc_scale_2 = nn.Linear(64 * 53 * 53, 128)
        self.fc_2_combined = nn.Linear(2 * 128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        scale_1 = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        scale_2 = F.max_pool2d(x, 2)
        scale_2 = self.dropout1(scale_2)
        scale_2 = torch.flatten(scale_2, 1)
        scale_2 = F.relu(self.fc_scale_2(scale_2))
        
        scale_1 = self.dropout1(scale_1)
        scale_1 = torch.flatten(scale_1, 1)
        scale_1 = F.relu(self.fc_scale_1(scale_1))

        combined_scales = torch.cat((scale_1, scale_2), dim=1)
        combined_scales = self.dropout2(combined_scales)
        combined_scales = self.fc_2_combined(combined_scales)
        output = F.log_softmax(combined_scales, dim=1)
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='Example of how a multi scale network benifits scale invarience.')
    parser.add_argument('--net-type', type=str, required=True, choices=['original', 'multi-scale'],
                        help='Pick the net type; original or multi-scale')    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True},)

    transform_resize = transforms.Compose([
        transforms.Resize(112),  # scale image to four times original
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    transform_pad = transforms.Compose([
        transforms.Pad(42),  # keep digits size the same but expand image
        transforms.RandomAffine(0, translate=(0.3, 0.3)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    resized_dataset_train = datasets.MNIST('../data', train=True, download=True,
                                    transform=transform_resize)
    padded_dataset_train = datasets.MNIST('../data', train=True, download=True,
                                 transform=transform_pad)
    dataset_train = ConcatDataset([resized_dataset_train, padded_dataset_train])
    train_loader = torch.utils.data.DataLoader(dataset_train, **kwargs)


    resized_dataset_test = datasets.MNIST('../data', train=False,
                                         transform=transform_resize)  
    padded_dataset_test = datasets.MNIST('../data', train=False,
                                      transform=transform_pad)
    dataset_test = ConcatDataset([resized_dataset_test, padded_dataset_test])
    test_loader = torch.utils.data.DataLoader(dataset_test, **kwargs)

    # display_sample_images(train_loader)

    if args.net_type == 'original':
        model = NetOrig().to(device)
    else:
        model = NetConcatAllScales().to(device)
    writer = SummaryWriter('runs/' + args.net_type + '-mnist')
    torch.onnx.export(model, torch.randn(1, 1, 112, 112).to(device),
                      args.net_type + '-model.onnx', output_names=['digits-class'])
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
