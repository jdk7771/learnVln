import random
import torch
from torch.utils import data

def synthetic_data(w, b, nums):
    x = torch.normal(0, 1, (nums,len(w)))
    y = torch.matmul(x, w)+b

    y += torch.normal(0, 0.01, y.shape)
    return x, y


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def data_iter(batchsize, feature, labels):
    num = len(feature)
    incides = list(range(num))

    random.shuffle(incides)

    for i in range(0, num, batchsize):
        batch_incides = torch.tensor(incides[i:min(i+ batchsize, num)])
        yield feature[batch_incides],labels[batch_incides]


def sgd(params ,lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -=lr*param.grad/ batch_size
            param.grad.zero_()


def linreg(x, w, b):
    return torch.matmul(x, w) + b

def train_model():
    num_epochs = 10
    lr = 0.03
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    print(w,b)
    true_w = torch.tensor([2, -3.4])
    true_b = torch.tensor([4.2])
    x,y = synthetic_data(true_w, true_b, 10000)
    for epoch in range(num_epochs):
        for x, y in data_iter(batchsize=50, feature=x,labels=y):
            l = squared_loss(linreg(x,w,b),y)
            l.sum().backward()
            sgd([w, b], lr, batch_size=30)
        with torch.no_grad():
            train1 = squared_loss(linreg(x,w,b),y)
            print(f'epoch {epoch + 1}, loss {float(train1.mean()):f}')
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')
    print(f'估计值为{w,b}')
