# %%
import torch.nn as nn
import torch
import torchvision
import time
from PIL import Image
from collections import OrderedDict
import logging
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR

# %%



# %%


# %%
logging.basicConfig(level=logging.DEBUG, filename='./log/sevenALEXNETcifar10400epoch.log',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')
print("three")
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
train_data = torchvision.datasets.CIFAR10('./data',
                            transform = trans,
                            train=True,
                            download = True)

# %%
test_data = torchvision.datasets.CIFAR10('./data',
                            transform = trans, train=False, download=True) 

# %%
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=200, shuffle=True, num_workers=2)

# %%
test_data_loader = torch.utils.data.DataLoader(test_data, 
                                               batch_size=200, 
                                               shuffle=False, 
                                               num_workers=2)

# %%


# %%

# %%
print(torch.cuda.is_available())

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',index= 2)
model = torchvision.models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[-1].out_features = 4
model = model.to(device)
loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)


# %%
def test():
    model.eval()
    test_acc_sum = 0.0    
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)


            test_acc_sum += (output.argmax(dim=1) == labels).float().sum().cuda(0).item()
 
        acc = test_acc_sum / len(test_data)
    return acc 


# %%
def training(curr_epoch):
    
    train_loss_sum = 0.0
    train_loss_sum2 = 0.0
    train_acc_sum = 0.0
    batch_count = 0
    model.train()
    for images, labels in train_data_loader:

        images = images.to(device)
        labels = labels.to(device)
        #print(images.shape, labels)
        
        #forward 
        output = model(images)

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
        train_loss_sum += loss.cuda(0).item()
            
        #print(output.argmax(dim=1) == labels)
        #print((output.argmax(dim=1) == labels).float())

        train_acc_sum += (output.argmax(dim=1) == labels).float().sum().cuda(0).item()
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
    
    epoch = 10
    for e in range(epoch):
        training(e)
    
    #save model weight
    torch.save(model.state_dict(), 'VGG.pkl')

# %%
if __name__ == '__main__':

    main()

# %%


# %%



