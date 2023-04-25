import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from model.model import YoloMulti
from bdd100k import BDD100k
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import SegmentationMetric
num = np.nan
print(5+num)
sys.exit()
ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]
device = torch.cuda.set_device(1)
metric = SegmentationMetric()

model = torch.load('model.pth')

transform = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
    ])
val_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=False, transform=transform, anchors=ANCHORS)
val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=16,
                                num_workers=12,
                                shuffle=False)

# Inference on validation for evaluation
mean_iou = [0] * 12
for _ in range(16):
    imgs, _, seg = next(iter(val_loader))
    imgs, seg = imgs.to(device), seg.to(device)  

    _, pseg = model(imgs)
    
    mean_iou += metric.mean_iou(seg, pseg)
print(mean_iou / 16)