import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf

import torchvision.datasets as ds
import torchvision.models as m
from torch.utils.data import DataLoader
from torch.utils.data import Subset

aug = tf.Compose([
    tf.RandomResizedCrop(32, scale=(0.8, 1.0)),
    tf.RandomHorizontalFlip(),
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def augmented_data(batch_size=4, subset_size=50):
    base_d = ds.CIFAR10(root='./data', train=True, download=True, transform=TCform(aug))
    sub_d = Subset(base_d, range(subset_size))  
    return DataLoader(sub_d, batch_size=batch_size, shuffle=True)

class TCform:
    def __init__(self, transf):
        self.transf = transf
        
        
    def __call__(self, x):
        return self.transf(x), self.transf(x) 


class S_Net(nn.Module):
    def __init__(self, out_d=128):
        super().__init__()
        
        base= m.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # 
        
        self.pj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,out_d)
            
        )
        
    def forward(self, x):
        h = self.encoder(x).squeeze() 
        z = self.pj(h)  
                
        return F.normalize(z, dim=1)

def loss_l(z_i, z_j, temp=0.5):
    
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim = sim / temp

    ls = torch.arange(N).to(z.device)
    ls = torch.cat([ls, ls], dim=0)
    mask = torch.eye(2*N).bool().to(z.device)
    
    sim.masked_fill_(mask, -9e15)

    t = torch.cat([torch.arange(N,2*N), torch.arange(0, N)]).to(z.device)
    loss = F.cross_entropy(sim, t)
    
    return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mod = S_Net().to(device)
opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
loader = augmented_data()  


def train(epochs=5):
    mod.train()
    for epoch in range(epochs):
        total_l = 0.0
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = mod(x1), mod(x2)
            
            loss = loss_l(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_l += loss.item()
            
            
        print(f"Epoch {epoch+1} Loss -> {total_l / len(loader):.4f}")





if __name__ == "__main__":
    print("Start training")
    train()
