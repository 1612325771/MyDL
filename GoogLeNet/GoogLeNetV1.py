# %%
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from PIL import Image
from collections import OrderedDict
import logging
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR
import os, glob
import csv
import pandas
import random
import torch
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image   #  pip install pillow
import numpy as np
from torchvision import transforms
import glob
import torchvision
from torch.utils.data import DataLoader
# %%
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# %%
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size = 1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size = 1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size = 3, padding = 1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# %%
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x

# %%
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

logging.basicConfig(level=logging.DEBUG, filename='../log/originGoogle.log',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')
print("three")
# %%
## read data
class_name_list = os.listdir("/home/bear/dl/data/dataset/animals")
species = ['elephant', 'giraffe', 'lion', 'monkey', 'tiger']
species_to_idx = dict((c, i) for i, c in enumerate(species))
image_dir = []
for species_name in class_name_list:
    image_dir += glob.glob(os.path.join('/home/bear/dl/data/dataset/animals', species_name, '*.jpg'))

##_________________________________makedataset______________________________________________
class makedataset(Dataset):
    def __init__(self, csv_filename, mode,transform):
        super(makedataset, self).__init__()
        self.csv_filename = csv_filename
        self.mode = mode
        self.transforms = transform
        self.image, self.label = self.load_csv()
        if mode == 'train':
            self.image = self.image[:int(0.8*len(self.image))]
            self.label = self.label[:int(0.8*len(self.label))]
        elif mode == 'val':
            self.image = self.image[int(0.8*len(self.image)):]
            self.label = self.label[int(0.8*len(self.label)):]
    def load_csv(self):
        image,label = [], []
        with open(self.csv_filename) as f:
            reader = csv.reader(f)
            for row in reader:
                i, l = row
                image.append(i)
                label.append(int(l))
        return image,label
    def __len__(self):
        return len(self.image)
    def __getitem__(self, index):
        img = self.image[index]
        label_tensor = torch.tensor(self.label[index])
        pil_img = Image.open(img)
        pil_img = pil_img.convert("RGB")
        data = self.transforms(pil_img)
        return data, label_tensor


# %%
# model = GoogLeNet(num_classes=10, aux_logits=True, init_weights=True)
model = torchvision.models.googlenet(pretrained=True)
model.fc.out_features = 10
# %%
trans = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


trans_test = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',index= 2)
# model = torchvision.models.vgg16(pretrained=True)
# model.classifier[-1].out_features = 4
model = model.to(device)
loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())



# %%
def test():
    model.eval()
    test_acc_sum = 0.0    
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)


            test_acc_sum += (output.argmax(dim=1) == labels).float().sum().cuda(2).item()
 
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
        logits = model(images)

        loss0 = loss_func(logits, labels)
        
        loss = loss0 
            
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
        train_loss_sum += loss.cuda(2).item()
            
        #print(output.argmax(dim=1) == labels)
        #print((output.argmax(dim=1) == labels).float())

        train_acc_sum += (logits.argmax(dim=1) == labels).float().sum().cuda(2).item()
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
    
    epoch = 30
    for e in range(epoch):
        training(e)
    
    #save model weight
    torch.save(model.state_dict(), '../checkpoints/originGoogle.pkl')

# %%
if __name__ == '__main__':

    main()

# %%


# %%
