# %%
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
import glob

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu',index= 2)

class animalModel(nn.Module):
    def __init__(self):
        super(animalModel, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        self.backbone = model
        self.fc1 = nn.Linear(512,5)
    def forward(self, x):
        out = self.backbone(x)
        out = self.fc1(out)
        return out
# %%
def predict(image):
    model = animalModel()
    model = model.to(device)
    output = model(image)
    
    ret_val, predicted = torch.max(output, 1)
    return predicted

# %%
all_imgs_path = glob.glob(r'/home/bear/dl/data/dataset/animal_test/*.jpg')

# %%
species = ['elephant', 'giraffe', 'lion', 'monkey', 'tiger']

# %%
idx_to_species = dict((v,k) for v, k in enumerate(species))
idx_to_species


# %%
trans = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

# %%


# %%
def recognition():
    print("begin to recogition")
    for img in all_imgs_path:
        for spec in species:
            if spec in img:
                test_img = Image.open(img)
                trans_img = trans(test_img).unsqueeze(0)
                input_img = trans_img.to(device)
                result = predict(input_img)
                result = result.item()
                print("real: "+spec + "  "+ "Predict: "+idx_to_species[result])

# %%
recognition()

# %%



