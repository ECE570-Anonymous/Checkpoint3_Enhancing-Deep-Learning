import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import torchvision.models as m


def cifar10_data(batch_size=8): 
    transform = tf.Compose([
        tf.Resize(32),
        tf.ToTensor(),
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    t_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    return DataLoader(t_set, batch_size=batch_size, shuffle=True)



class S_Loss(nn.Module):
    def __init__(self, bt=0.9):
        super().__init__()
        self.bt = bt

    def forward(self, lg, tg):
        probs = F.softmax(lg, dim=1)
        one_hot = F.one_hot(tg, num_classes=probs.size(1)).float()
        
        sm = self.bt * one_hot + (1 - self.bt)* probs
        log_p = F.log_softmax(lg, dim=1)
        
        
        return -(sm * log_p).sum(dim=1).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 5
learning_rate = 0.001

model = m.mobilenet_v2(weights=None, num_classes=10).to(device)
crit = S_Loss(bt=0.9)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_l = cifar10_data(batch_size)

def train(): 
    for epoch in range(num_epochs):

        # if epoch == 0:
        #     print(outputs.shape)
        running_loss = 0.0
        correct, total = 0, 0
        
        for i, l in train_l:
            i, l = i.to(device), l.to(device)
            opt.zero_grad()
            outputs = model(i)
            loss = crit(outputs, l)
            loss.backward()
            
            opt.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            total += l.size(0)
            correct += (predicted == l).sum().item()
        epoch_l = running_loss/len(train_l)
        epoch_a = 100 * correct / total
        print(f"Epoch {epoch+1} , Loss: {epoch_l:.4f} , Accuracy: {epoch_a:.2f}%")

if __name__ == "__main__":
    print("Start training")
    train()
