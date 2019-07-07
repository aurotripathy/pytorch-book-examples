import csv
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def get_dataset_labels():
    dataset = []
    labels = []
    with open('io/dataset.csv', 'r') as f:
        rows = csv.reader(f, delimiter=',')
        for row in rows:
            labels.append(int(row[2]) - 1)  # convert label from 1 and 2 to 0 and 1
            dataset.append((float(row[0]), float(row[1])))
    return dataset, labels


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def display(Xs, ys, title):
    category_1_x1 = []
    category_1_x2 = []
    category_2_x1 = []
    category_2_x2 = []

    for X, y in zip(Xs, ys):
        if y == 0:
            category_1_x1.append(X[0])
            category_1_x2.append(X[1])
        else:
            category_2_x1.append(X[0])
            category_2_x2.append(X[1])

    plt.clf()
    plt.scatter(category_1_x1, category_1_x2, marker='^', c='lightgreen', label='cat_1')
    plt.scatter(category_2_x1, category_2_x2, marker='o', c='lightblue', label='cat_2')

    plt.title(title)
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(os.path.join('io', title))


def display_misclassified(Xs, ys_gt, ys_preds, title):
    category_1_x1 = []
    category_1_x2 = []
    category_2_x1 = []
    category_2_x2 = []
    category_miss_class_x1 = []
    category_miss_class_x2 = []
    
    for X, y_gt, y_pred in zip(Xs, ys_gt, ys_preds):
        if y_gt == 0:
            category_1_x1.append(X[0])
            category_1_x2.append(X[1])
        else:
            category_2_x1.append(X[0])
            category_2_x2.append(X[1])
        if y_gt != y_pred:
            category_miss_class_x1.append(X[0])
            category_miss_class_x2.append(X[1])

    plt.clf()
    plt.scatter(category_1_x1, category_1_x2, marker='^', c='lightgreen', label='cat_1')
    plt.scatter(category_2_x1, category_2_x2, marker='o', c='lightblue', label='cat_2')
    plt.scatter(category_miss_class_x1, category_miss_class_x2,
                s=80, facecolors='none', edgecolors='r', label='miss')


    plt.title(title)
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(os.path.join('io', title))
