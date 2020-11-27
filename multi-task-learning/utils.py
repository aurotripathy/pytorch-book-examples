import matplotlib.pyplot as plt
from pudb import set_trace

def normalize(X):
    print('mean:', X.mean())
    print('std:', X.std())
    return (X - X.mean()) / X.std()


def display_sample_images(train_loader, count=3):
    # Pick first sample in each mini-batch.
    for i, (data, target) in enumerate(train_loader):
        print('Showing label:', target)
        plt.imshow(data[0][0].numpy())  # data is [batch]x[channel]x[H]x[W}]
        plt.show()
        if i == count-1:
            break
