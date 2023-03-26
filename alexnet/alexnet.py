# %%
import torch.nn as nn
import torch
import torchvision
import time
from PIL import Image
from collections import OrderedDict
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR

# %%
class FcNet3(nn.Module):
    
    def __init__(self, **kwargs):
        super(FcNet3, self).__init__(**kwargs)
        
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
    
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
 )
    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


# %%
fc = FcNet3()

# %%


# %%


# %%


# %%
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224,224]),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor()
])

# %%
train_data = torchvision.datasets.CIFAR10('../data',
                            transform = trans,
                            train=True,
                            download = True)

# %%
test_data = torchvision.datasets.CIFAR10('../data',
                            transform = trans, train=False, download=True) 

# %%
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

# %%
test_data_loader = torch.utils.data.DataLoader(test_data, 
                                               batch_size=32, 
                                               shuffle=False, 
                                               num_workers=2)

# %%
loss_func = torch.nn.CrossEntropyLoss()

# %%
optimizer = torch.optim.Adam(fc.parameters())

# %%
print(torch.cuda.is_available())

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',index= 3)
fc = fc.to(device)
optimizer = torch.optim.SGD(fc.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)


# %%
def test():
    test_acc_sum = 0.0    
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = fc(images)

            test_acc_sum += (output.argmax(dim=1) == labels).float().sum().cuda(3).item()
 
        acc = test_acc_sum / len(test_data)
    return acc 


# %%
def training(curr_epoch):
    
    train_loss_sum = 0.0
    train_loss_sum2 = 0.0
    train_acc_sum = 0.0
    batch_count = 0

    for images, labels in train_data_loader:

        images = images.to(device)
        labels = labels.to(device)
        #print(images.shape, labels)
        
        #forward 
        output = fc(images)

        loss = loss_func(output, labels)
            
        #gradient clear 
        optimizer.zero_grad()

        #backward 
        loss.backward()
            
        #update weight
        optimizer.step()
        scheduler.step()
            
        #train_loss_sum += loss.item()
        #GPU 
        #item() convert Tersor to Python number
        # 
        train_loss_sum += loss.cuda(3).item()
            
        #print(output.argmax(dim=1) == labels)
        #print((output.argmax(dim=1) == labels).float())

        train_acc_sum += (output.argmax(dim=1) == labels).float().sum().cuda(3).item()
        batch_count += 1
        
    #print(train_acc_sum) 
    train_acc = train_acc_sum / len(train_data)
    batch_avg_loss = train_loss_sum / batch_count
    
    test_acc = test()
    logging.info("epoch %d, loss %.4f, train accuracy %.3f, test accuracy %.3f" %
         (curr_epoch, batch_avg_loss, train_acc, test_acc))
    print("epoch %d, loss %.4f, train accuracy %.3f, test accuracy %.3f" %
         (curr_epoch, batch_avg_loss, train_acc, test_acc))

# %%
def main():
    
    epoch = 400
    for e in range(epoch):
        training(e)
    
    #save model weigh

# %%
if __name__ == '__main__':

    main()

# %%


# %%



