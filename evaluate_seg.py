import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.model import YoloMulti
from bdd100k import BDD100k
import torch.utils.data as data
import torchvision.transforms as transforms

ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]


model = torch.load('model.pth')

transform = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
    ])
val_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=False, transform=transform, anchors=ANCHORS)
val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=16,
                                num_workers=12,
                                shuffle=False)