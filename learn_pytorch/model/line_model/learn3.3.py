from learn import *
import torch
from torch.utils import data

def load_array(data_array, batch_size, is_train=True):
    data_set = data.TensorDataset(*data_array)
    return data.DataLoader(data_set, batch_size, shuffle= is_train)


true_w = torch.tensor([3.5, 4.5])
true_b = torch.tensor(4)
features, labels = synthetic_data(true_w, true_b, 100)
data_iter = load_array((features, labels), 20, True)

print(next(iter(data_iter)))

net = torch.nn.Sequential

