from softmax_nodel import *
from torch import nn

batch_size = 256

train_iter = get_iter(256)
test_iter = get_test_iter(256)


net = nn.Sequential(nn.Flatten(),nn.Linear(784, 10))

def init_weight(Layer):
    if isinstance(Layer, nn.Linear):
        nn.init.normal_(Layer.weight, 0)
        if Layer.bias != 0:
            nn.init.constant_(Layer.bias, 0)


def train(net, train, test, loss, num_epochs, trainer):
    for x,y in train:
        y_hat = net
net.apply(init_weight)

loss = nn.CrossEntropyLoss()

trainer = torch.optim.sgd(net.parameters, lr = 0.01)

