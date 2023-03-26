# %%
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import time
from PIL import Image
from collections import OrderedDict
import logging
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR

# %%



# %%


# %%
logging.basicConfig(level=logging.DEBUG, filename='../log/animalVGG.log',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')
print("three")
# %%


# %%
img_dir = r'/home/bear/dl/data/dataset/animals'


# %%
trans = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
    torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])
dataset =  torchvision.datasets.ImageFolder(
        img_dir,
        transform=trans
)
count = len(dataset)
print(count)
train_count = int(0.8*count)
test_count = count - train_count
train_dataset, test_dataset = data.random_split(dataset, [train_count, test_count])



# %%
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

# %%
test_data_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=32, 
                                               shuffle=False, 
                                               num_workers=2)

# %%


# %%

# %%
print(torch.cuda.is_available())

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',index= 2)
model = torchvision.models.vgg16(pretrained=True)
model.classifier[-1].out_features = 5
model = model.to(device)
loss_func = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)



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
 
        acc = test_acc_sum / len(test_dataset)
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
        train_loss_sum += loss.cuda(2).item()
            
        #print(output.argmax(dim=1) == labels)
        #print((output.argmax(dim=1) == labels).float())

        train_acc_sum += (output.argmax(dim=1) == labels).float().sum().cuda(2).item()
        batch_count += 1
        
    #print(train_acc_sum) 
    train_acc = train_acc_sum / len(train_dataset)
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
    torch.save(model.state_dict(), 'fourVGG.pkl')

# %%
if __name__ == '__main__':

    main()

# %%


# %%



