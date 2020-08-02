class UnNormalize(object):
    """ The purpose of this class is to display the augmented image"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def display_sample_images(train_loader):
    for data, _ in train_loader:
        unorm = UnNormalize(mean=(0.1307,), std=(0.3081,))
        img = unorm(data[0])
        plt.imshow(img[0].numpy())
        plt.show()
