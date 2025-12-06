import torch
import torch.nn as nn
from torch.nn import functional as F
import os

current_dir = os.getcwd()
batch_size = 5
max_pre_cha = 200
block_size = 6
train_iter = 3000
val_iter = 30

device = "cuda" if torch.cuda.is_available() else "cpu"
with open("input.txt", 'r') as f:
    text = f.read()

#construct a mapping table and data to train and validation

chars = sorted(list(set(text)))
var_size = len(chars)


stoi = {cha:i for i,cha in enumerate(chars)}
itos = {i:cha for i,cha in enumerate(chars)}
encoder = lambda char :[stoi[cha] for cha in char]
decoder = lambda inter : ''.join([itos[i] for i in inter])

text = encoder(text)
train_data = text[:int(0.9*len(text))]
val_data = text[int(0.9*len(text)):]


train_data = torch.tensor(train_data)
val_data = torch.tensor(val_data)


def get_batch(spilt):

    data = train_data if spilt == "train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x =  torch.stack([data[i:i+block_size] for i in ix] )
    y =  torch.stack([data[i+1:i+block_size+1] for i in ix] )
    x,y = x.to(device),y.to(device)

    return x,y

@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()

    losses = torch.zeros(val_iter)
    for spilt in ["train","val"]:
        for i in range(val_iter):
            x,y = get_batch(spilt)
            logit,loss = model(x,y)
            losses[i] = loss.item()
        output[spilt] = torch.cat((torch.tensor(losses.mean()).unsqueeze(0),losses))

    return output


        
class BigramLanuageModel(nn.Module):
    ### input idx dimotion B   T
    def __init__(self, idx):
        super().__init__()
        self.embedding = nn.Embedding(var_size,var_size)

    def forward(self, idx, target =None):
        logits = self.embedding(idx)
        if target == None:
            loss = None
        else:
            ### The tensor is in channels-first format with shape (batch, channels, height, width).
            B, T, C = logits.shape
            logits = logits.view((B*T,C))
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits,loss

    def generate(self, idx ,max_pre_cha):
        for i in range(max_pre_cha):
            logits,_ = self(idx)
            logits = logits[:,-1,:]
            logits =F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(logits, num_samples=1)
            idx = torch.cat((idx, idx_next),1)

        return idx
    
model = BigramLanuageModel(var_size)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),1e-2)
def train():
    for i in range(train_iter):
        optimizer.zero_grad()
        x,y = get_batch('train')
        logits,loss = model(x,y)
        loss.backward()
        optimizer.step()
        if ((i%30) == 0):
            print(f"第{i}次训练的损失为{estimate_loss()['train'][0]}")
    torch.save(model.state_dict(), os.path.join(current_dir, 'bigram_model.pth'))

def test():
    model.load_state_dict(torch.load('bigram_model.pth'))
    model.eval()
    context = torch.zeros((1,1),dtype = torch.long, device=device)
    print(decoder(model.generate(context,max_pre_cha)[0].tolist()))

test()
