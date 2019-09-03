import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim, xlim, axhline, axvline

import numpy as np
import os

def display_loss(loss, title, folder='plots'):

    if not os.path.exists(folder):
        os.makedirs(folder)

    x_axis = np.arange(len(loss))
    y_axis = loss

    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis)

    ax.set(xlabel='Epochs', ylabel='Epoch Loss',
           title=title)
    ax.grid()

    fig.savefig(os.path.join(folder, title + '.png'))
    plt.show()



def display_points(Xs, ys, title, folder='plots'):
    if not os.path.exists(folder):
        os.makedirs(folder)

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
    ylim((-3, 3))
    xlim((-3, 3))
    plt.grid(True)
    axhline(linewidth=2, color='k') #adds thick black line @ y=0
    axvline(linewidth=2, color='k') #adds thick black line @ x=0
    plt.scatter(category_1_x1, category_1_x2, marker='o', c='red', label='cat_1')
    plt.scatter(category_2_x1, category_2_x2, marker='s', c='green', label='cat_2')

    plt.title(title)
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(folder, title + ".png"))

def plot_points_line_slope_intercept(Xs, ys, slope,
                                     intercept, title,
                                     x_range=(-3, 3), folder='plots'):
    """Plot points and line from slope and intercept"""

    if not os.path.exists(folder):
        os.makedirs(folder)

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
    ylim((-3, 3))
    xlim((-3, 3))
    plt.grid(True)
    axhline(linewidth=2, color='k') #adds thick black line @ y=0
    axvline(linewidth=2, color='k') #adds thick black line @ x=0
    plt.scatter(category_1_x1, category_1_x2, marker='o', c='red', label='cat_1')
    plt.scatter(category_2_x1, category_2_x2, marker='s', c='green', label='cat_2')

    plt.title(title)
    plt.legend(loc='upper right')


    x_vals = np.arange(x_range[0], x_range[1] + 1, 1)
    y_vals = intercept + slope * x_vals
    plt.grid(True)
    axhline(linewidth=2, color='k') #adds thick black line @ y=0
    axvline(linewidth=2, color='k') #adds thick black line @ x=0
    ylim((x_range[0], x_range[1]))
    xlim((x_range[0], x_range[1]))
    plt.plot(x_vals, y_vals, '-')
    # plt.show()
    plt.savefig(os.path.join(folder, title + ".png"))
    plt.close()
    
if __name__ == '__main__':
    plot_line_from_slope_intercept(0.5,  0)
