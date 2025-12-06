from Learn_ros.src.learn_pytorch.utlis.image_load import get_mnist_label, show_mnist_image
from torchvision import transforms,datasets
from torch import utils
from torch.utils import data
import torchvision
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utlis.utlis import *
def get_iter_numswork():
    return 4


def get_iter(batchsize):
    trans = transforms.ToTensor()

    #？？？？？？？？？？？？？这个如何直接读取
    mnist_train = torchvision.datasets.FashionMNIST(root = "../data", transform=trans, train = True, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root = "../data", transform=trans, train = False, download=True)
    return  data.DataLoader(mnist_train , batch_size=batchsize, shuffle=True, num_workers=get_iter_numswork())
def get_test_iter(batchsize):
    trans = transforms.ToTensor()

    mnist_test = torchvision.datasets.FashionMNIST(root = "../data", transform=trans, train = False, download=True)
    return  data.DataLoader(mnist_test , batch_size=batchsize, shuffle=True, num_workers=get_iter_numswork())
def sgd(w, lr, batch):
    with torch.no_grad():
        for wsin in w:
            wsin-=wsin.grad*lr/batch
            wsin.grad.zero_()

def softmax(result):
    result = torch.exp(result)
    result = result/torch.sum(result, dim=1, keepdim=True)
    return result

"""feature:batchsize*feature
    w:feature*10
    b:10*1
    return B*10
"""
def line(feature, w, b):
    result = torch.matmul(feature.flatten(1),w)+b
    return result


def net(feature, w, b):
    result = line(feature, w, b)
    return softmax(result)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y ])

def get_resultpre(y_hat):
    y_hat, y_ind = y_hat.max(dim =1) 
    return y_ind


def train():

    accurary = Get_accuracy()

    batchsize = 64
    nums_chara = 28*28
    numepoch = 12

    w = torch.normal(0, 0.01, size=(nums_chara, 10), requires_grad=True)
    b = torch.zeros((10), requires_grad=True)

    for epoch in range(numepoch):
        for feartures,labels in get_iter(batchsize):
            yhat = net(feartures, w, b)
            loss = cross_entropy(yhat, labels)
            loss.sum().backward()
            sgd((w,b),lr=0.003,batch=batchsize)
            accurary.update(get_resultpre(yhat), labels)
        print(f"第{epoch}轮准确率为{accurary.get_acc()}")
        accurary.clear()
    torch.save(
        {
            'w':w,
            'b':b
        }, 'softmax.pth'
    )

def test(model_path = 'softmax.pth'):
    checkpoint = torch.load(model_path)
    w = checkpoint['w']
    b = checkpoint['b']
    test_loader = get_test_iter(batchsize=256)
    print("模型导入成功")
    accuracy = Get_accuracy()
    with torch.no_grad():
        for x,y in test_loader:
            y_hat = net(x,w,b)
            y_inx = get_resultpre(y_hat)
            accuracy.update(y_inx, y)
    test_acc = accuracy.get_acc()
    print(f"测试集准确率: {test_acc:.4f}")
    
test('softmax.pth')      





