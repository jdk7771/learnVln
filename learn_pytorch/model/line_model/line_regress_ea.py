from line_regress_base import *
from torch.utils import data
from torch import nn


def  load_array(array, batch_size, is_train = True):
    dataset = data.TensorDataset(*array)
    return data.DataLoader(dataset, shuffle = is_train, batch_size=batch_size)



def train():

    w = torch.normal(0, 0.02, size=(3,1), requires_grad=True)
    b = torch.normal(0, 0.02, size=(1,), requires_grad=True)
    true_w = torch.tensor([2.1, 3.8, 4.3]).reshape(w.shape)
    true_b = torch.tensor([3.3])

    x, y = generate_line_data(true_w, true_b, 1000)

    epoch = 10
    batch_size = 30
    lr = 0.01

    net = nn.Sequential(nn.Linear(3,1))
    net[0].weight.data.normal_(0, 0.1)
    net[0].bias.data.fill_(0)

    trainer = torch.optim.SGD(net.parameters(), lr = lr)
    loss = nn.MSELoss()

    
    for i in range(epoch):
        for feature ,lables in load_array((x,y), batch_size):
            l = loss(net(feature), lables)
            trainer.zero_grad()
            l.backward()
            trainer.step()

    fitted_weight = net[0].weight.data
    fitted_bias = net[0].bias.data
    
    print(f"真实权重: {true_w.flatten()}")
    print(f"拟合权重: {fitted_weight.flatten()}")
    print(f"真实偏置: {true_b.item()}")
    print(f"拟合偏置: {fitted_bias.item():.4f}")

train()


