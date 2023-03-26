# %%
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import time
from PIL import Image
from collections import OrderedDict
import logging
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR

# %%
class vgg(nn.Module):
    
    def __init__(self, **kwargs):
        super(vgg, self).__init__(**kwargs)
        
        self.convnet = nn.Sequential(
           #block1
           nn.Conv2d(in_channels=3, out_channels=64,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=64, out_channels=64,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           #block2
           nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           #block3
           nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           #block4
           nn.Conv2d(in_channels=256, out_channels=512,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           #block5
           nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(7, 7))
        )
    
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    def forward(self, img):
        output = self.convnet(img)
        output = self.avgpool(output)
        output = torch.flatten(output,1)
        output = self.fc(output)
        return output


# %%


# %%
logging.basicConfig(level=logging.DEBUG, filename='../log/originVGG.log',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')
print("three")
# %%


# %%
model = vgg()

# %%
trans = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    transforms.ToTensor()
])


trans_test = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor()
])
# %%
train_data = torchvision.datasets.CIFAR10('../data',
                            transform = trans,
                            train=True,
                            download = True)

# %%
test_data = torchvision.datasets.CIFAR10('../data',
                            transform = trans_test, train=False, download=True) 

# %%
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

# %%
test_data_loader = torch.utils.data.DataLoader(test_data, 
                                               batch_size=32, 
                                               shuffle=False, 
                                               num_workers=2)

# %%


# %%

# %%
print(torch.cuda.is_available())

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',index= 3)
# model = torchvision.models.vgg16(pretrained=True)
# model.classifier[-1].out_features = 4
model = model.to(device)
loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)



# %%
def test():
    model.eval()
    test_acc_sum = 0.0    
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)


            test_acc_sum += (output.argmax(dim=1) == labels).float().sum().cuda(3).item()
 
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
    
    epoch = 20
    for e in range(epoch):
        training(e)
    
    #save model weight
    torch.save(model.state_dict(), '../checkpoints/originVGG2.pkl')

# %%
if __name__ == '__main__':

    main()

# %%


# %%



