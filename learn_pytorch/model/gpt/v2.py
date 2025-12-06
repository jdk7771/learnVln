#!/usr/bin/env python
# coding: utf-8
######################################################################33

##最好修改为动态的输入和输出


# In[32]:


import torch
import torch.nn as nn
from torch.nn import functional as F
import os


# In[33]:


current_dir = os.getcwd()
batch_size = 32
max_pre_cha = 800
block_size = 30
train_iter = 300
val_iter = 30
n_embed = 256
n_heads = 8
n_layer = 6
dropout = 0.2

# In[34]:


device = "cuda" if torch.cuda.is_available() else "cpu"
with open("input.txt", 'r') as f:
    text = f.read()


# onstruct a mapping table and data to train and validation

# In[35]:


chars = sorted(list(set(text)))
var_size = len(chars)


# In[36]:


stoi = {cha:i for i,cha in enumerate(chars)}
itos = {i:cha for i,cha in enumerate(chars)}
encoder = lambda char :[stoi[cha] for cha in char]
decoder = lambda inter : ''.join([itos[i] for i in inter])


# In[37]:


text = encoder(text)
train_data = text[:int(0.9*len(text))]
val_data = text[int(0.9*len(text)):]


# In[38]:


train_data = torch.tensor(train_data)
val_data = torch.tensor(val_data)


# In[39]:


def get_batch(spilt):
    data = train_data if spilt == "train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x =  torch.stack([data[i:i+block_size] for i in ix] )
    y =  torch.stack([data[i+1:i+block_size+1] for i in ix] )
    x,y = x.to(device),y.to(device)
    return x,y


# In[40]:


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


# In[41]:


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size,bias=None)
        self.query = nn.Linear(n_embed, head_size, bias=None)
        self.value = nn.Linear(n_embed, head_size, bias=None)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        
    def forward(self, idx):
        B,T,C = idx.shape
        #BTC->BTT
        k = self.key(idx)
        q = self.query(idx)
        v = self.value(idx)

        wei = q @ k.transpose(-1,-2) * k.shape[-1]**-0.5
        mask_wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        wei = F.softmax(mask_wei,dim=-1)
        out = wei@v

        return out


# In[42]:


class MultiHeadAttention(nn.Module):
    def __init__(self,num_head, head_size):
        super().__init__()
        self.head = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(num_head*head_size ,n_embed)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.head], dim=-1)
        out = self.proj(out)
        return out


# In[43]:


class FeedForward(nn.Module):
    def __init__(self,n_embed ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_embed, n_embed)
        )
    def forward(self, x):
        return self.net(x)


# In[44]:


class Block(nn.Module):
    def __init__(self, num_heads, num_embed):
        super().__init__()
        self.multi_headatt = MultiHeadAttention(num_heads, num_embed//4)
        self.ffwd = FeedForward(n_embed)
        self.lanor1  = nn.LayerNorm(n_embed)
        self.lanor2  = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.multi_headatt(self.lanor1(x))
        x = x + self.ffwd(self.lanor2(x))

        return x 

# In[45]:


class GPTLanuageModel(nn.Module):
    ### input idx dimotion B   T
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(var_size,n_embed)
        self.position_embedding = nn.Embedding(block_size,n_embed)
        self.block = nn.Sequential(*[Block(n_heads, n_embed) for _ in range(n_layer)])
        self.linear = nn.Linear(n_embed, n_embed)
        self.linee = nn.Linear(n_embed, var_size)
    def forward(self, idx, target =None):
        B,T = idx.shape
        input(idx)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(0,T,device=device))

        x = tok_emb + pos_emb
        input(x.shape)
        x = self.block(x)
        x = self.linear(x)
        logits = self.linee(x)

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

            idx_cond = idx[:,-block_size:]
            logits,_ = self(idx_cond)
            logits = logits[:,-1,:]
            logits =F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(logits, num_samples=1)
            idx = torch.cat((idx, idx_next),1)
        return idx
    


# In[46]:


model = GPTLanuageModel()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),3e-3)
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


# In[ ]:


def test():
    model.load_state_dict(torch.load('bigram_model.pth'))
    model.eval()
    context = torch.zeros((1,1),dtype = torch.long, device=device)
    print(decoder(model.generate(context,max_pre_cha)[0].tolist()))


# In[ ]:

import argparse

def main():
    parser = argparse.ArgumentParser(description='GPT Language Model')
    parser.add_argument('mode', choices=['train', 'test'], help='运行模式: train 或 test')
    parser.add_argument('--model_path', default='bigram_model.pth', help='模型路径')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()

if __name__ == "__main__":
    main()
