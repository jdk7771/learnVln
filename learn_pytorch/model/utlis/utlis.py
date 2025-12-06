# import numpy 
# import torch

# class Get_accuracy():
#     def __init__(self):
#         self.label_pre = []
#         self.label_real = []

#     def update(self,y_hat,y):
#         if torch.is_tensor(y_hat):
#             y_hat = y_hat.detach().cpu().numpy()
#         if torch.is_tensor(y):
#             y = y.detach().cpu().numpy()
#         self.label_pre.append(y_hat)
#         self.label_real.append(y)

#     def get_acc(self):
#         if(self.label_real == []):
#             print("没有数据无准确率")
#             return 0
#         cmd = self.label_pre == self.label_real
#         return cmd.sum()/len(self.label_real)
    
#     def clear(self):
#         self.label_pre = []
#         self.label_real = []

import torch

class Get_accuracy():
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, y_hat, y):
        # 完全使用PyTorch，避免numpy转换问题
        self.correct += (y_hat == y).sum().item()
        self.total += y.shape[0]

    def get_acc(self):
        if self.total == 0:
            print("没有数据无准确率")
            return 0
        print(f"总数为{self.total}")
        return self.correct / self.total
    
    def clear(self):
        self.correct = 0
        self.total = 0


def test(test_iter,net, device, loss):
    if device is None:
        device = next(iter(test_iter)).device

    net.eval()
    accuracy = Get_accuracy()

    with torch.no_grad():
        for x, y in test_iter:
            y_hat = net(x)
            los = loss(y_hat, y )
            accuracy.update(y_hat, y )
            total_loss += los.item()

        accuracy_da = accuracy.get_acc()
        avg_loss = total_loss / len(test_iter)    

    print(f'测试集 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}')
    return accuracy, avg_loss



def train(net, train_data, test_data, loss, num_epochs, trainer, device):
    if device is None:
        device = next(net.parameters()).device

    net.to(device)
    for i in range(num_epochs):
        net.train()
        for x,y in train_data:
            y_hat = net(x)
            trainer.zero_grad()
            loss.backward()
            trainer.step()

    test(test_data,net, loss)

