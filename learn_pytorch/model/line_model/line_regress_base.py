##定义线性回归模型的训练过程
#生成数据训练数据-
#epoch优化（batchsize 优化 做一个iter）
#iter（计算loss-loss反向-参数优化（中间需要停止grad））
import torch
import random
from matplotlib import pyplot as plt

def generate_line_data(w, b, nums):
    x = torch.normal(0, 1, (nums, len(w)))
    y = torch.matmul(x, w) + b 
    y += torch.normal(0, 0.01,size=y.shape)

    return x, y 
def data_iter(batch_size, x, y):
    num = len(x)
    incides = list(range(num))
    random.shuffle(incides)
    for i in range(0, num, batch_size):
        batch_incides = incides[i:min(i+batch_size, num)]
        yield x[batch_incides],y[batch_incides]

def squared_loss(y_hat, y):
    return (y_hat-y)**2/2

def linereg(x, w, b):
    return torch.matmul(x,w)+b

def sgd(w, lr, batch_size):
    with torch.no_grad():
        w -= w.grad / batch_size * lr
        w.grad.zero_()

def train():
    num_epochs = 30
    lr = 0.01
    batch_size = 50


    w = torch.normal(0, 0.02, size=(3,1), requires_grad=True)
    b = torch.normal(0, 0.02, size=(1,), requires_grad=True)

    true_w = torch.tensor([2.1, 3.8, 4.3]).reshape(w.shape)
    true_b = torch.tensor([3.3])

    x, y = generate_line_data(true_w, true_b, 1000)
    plt.plot(x, y, "ro", markersize = 3)
    plt.show()
    for epoch in range(num_epochs):
        for feature, lable in data_iter(batch_size, x, y):
            y_hat = linereg(feature, w, b)
            loss = squared_loss(y_hat, lable)   
            loss.sum().backward()
            sgd(w, lr, batch_size)
            sgd(b, lr, batch_size)
        with torch.no_grad():
            print(f'loss：{loss} epoch:{epoch}')
    
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')
    print(f'估计值为{w,b}')

