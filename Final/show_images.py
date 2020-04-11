import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img,c=""):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(7,7))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(c)

def show_random_images(dataset, classes):
    # get some random training images
    #dataiter = iter(dataset)
    #images, labels = dataiter.next()

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    print(dataset)
    dataiter = iter(dataset)
    images, labels = dataiter.next()
    for i in range(10):
        index = [j for j in range(len(labels)) if labels[j] == i]
        imshow(torchvision.utils.make_grid(images[index[0:5]],nrow=5,padding=2,scale_each=True),classes[i])