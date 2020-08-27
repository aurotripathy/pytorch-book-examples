# https://theaisummer.com/receptive-field/

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
import matplotlib.pyplot as plt
from pudb import set_trace


class NetTwoScalesMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_t = nn.Conv2d(1, 32, 3, 1)
        self.conv2_t = nn.Conv2d(32, 64, 3, 1)
        self.conv3_t = nn.Conv2d(64, 128, 3, 1)
        self.conv4_t = nn.Conv2d(128, 256, 3, 1)

        self.fc1_t = nn.Linear(256 * 5 * 5, 128)
        self.fc1_b = nn.Linear(64 * 26 * 26, 128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc2 = nn.Linear(2 * 128, 10)


    def forward(self, x):
        x_t = self.pool(F.relu(self.conv1_t(x)))
        x_t_b = self.pool(F.relu(self.conv2_t(x_t)))
        x_t = self.pool(F.relu(self.conv3_t(x_t_b)))
        x_t = self.pool(F.relu(self.conv4_t(x_t)))
        x_t = x_t.view(-1, 256 * 5 * 5)
        x_t = self.fc1_t(x_t)

        x_b = x_t_b.view(-1, 64 * 26 * 26)
        x_b = self.fc1_b(x_b)
       
        x = torch.cat((x_t, x_b), dim=1)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

        
class NetTwoStream(nn.Module):
    """ Two parallel streams, top, bottom concatenateed in the end """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 26 * 26, 128)
        self.fc2 = nn.Linear(2 * 128, 10)


    def forward(self, x):
        x_top = self.pool(F.relu(self.conv1(x)))
        x_bot = self.pool(F.relu(self.conv1(x)))

        x_top = self.pool(F.relu(self.conv2(x_top)))
        x_bot = self.pool(F.relu(self.conv2(x_bot)))

        x_top = x_top.view(-1, 64 * 26 * 26)
        x_top = self.fc1(x_top)

        x_bot = x_bot.view(-1, 64 * 26 * 26)
        x_bot = self.fc1(x_bot)

        x = torch.cat((x_top, x_bot), dim=1)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
class NetConcatAllScales(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, dilation=2)
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


class NetOrig(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform_resize = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    transform_pad = transforms.Compose([
        transforms.Pad(42),
        transforms.RandomAffine(0, translate=(0.3, 0.3)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset_resize = datasets.MNIST('../data', train=True, download=True,
                                    transform=transform_resize)
    dataset_pad = datasets.MNIST('../data', train=True, download=True,
                                 transform=transform_pad)
    dataset_train = ConcatDataset([dataset_resize, dataset_pad])
    train_loader = torch.utils.data.DataLoader(dataset_train, **kwargs)


    dataset_resize_test = datasets.MNIST('../data', train=False,
                                         transform=transform_resize)  
    dataset_pad_test = datasets.MNIST('../data', train=False,
                                      transform=transform_pad)
    dataset_test = ConcatDataset([dataset_resize_test, dataset_pad_test])
    test_loader = torch.utils.data.DataLoader(dataset_test, **kwargs)

    # display_sample_images(train_loader)
    
    # model = NetTwoScalesMnist().to(device)
    # model = NetOrig().to(device)
    model = NetConcatAllScales().to(device)
    print(model)
    print('Paramter count in Model:', sum([torch.numel(p) for p in model.parameters()]))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
