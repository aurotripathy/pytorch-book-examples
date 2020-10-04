
This is a simple example of how to design a neural network that learns to classify images at two different scale. The original MNIST dataset has been used to illustrate how well convolutional neural networks (CNNs) extract features and how we can build a decision boundary to classify digits. 

This example uses an extended MNIST dataset with two different scales of digits. We create this programatically.

First
```
    transform_resize = transforms.Compose([
        transforms.Resize(112),  # scale image to four times original
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
```

Second


### Sample Images
These sample images are at two different scales.

| Scaled Image                                              | Original Image with rest of area padded                   |
| ----------------------------------------------------------|---------------------------------------------------------- |
|![mnist images](./assets/Figure_1.png "Image") | ![mnist images](./assets/Figure_2.png "Image")|


### The Two Nets
![mnist images](./assets/combined-nets.png "Image")


### TensorBoard Plots
![mnist images](./assets/TB-test-accuracy-loss.PNG "Image")
