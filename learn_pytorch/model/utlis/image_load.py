import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

def get_mnist_label(label):
    mnist_label = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return[mnist_label[i] for i in label]

def show_mnist_image(imgs, nums_row, nums_line, scale=1.5, titles=None):
    figsize = (nums_row*scale, nums_line*1.5)
    _, axes = plt.subplots(nums_row, nums_line, figsize = figsize)
    axes = axes.flatten()
    for i,(ax, image) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(image):
            ax.imshow(image.squeeze(0).numpy())#维度问题可能会出现
        else:
            ax.imshow(image)
        if titles!=None:
            ax.set_title(titles[i])
    return axes

# trans = transforms.ToTensor()

# mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True )
# mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True )
# print(next(iter(data.DataLoader(mnist_train , batch_size=5, shuffle=True, num_workers=4))))


# img,lab = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_mnist_image(img, 2, 9,titles= get_mnist_label(lab))
# plt.show()