import argparse

import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from model import YoloMulti
from bdd100k import BDD100k
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss, SegmentationLoss, DetectionLoss

import matplotlib.pyplot as plt
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/prototype_lane');

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.cuda.set_device(1)
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

'''
Author: Pume Tuchinda
'''


ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers')
    # Fix root below
    parser.add_argument('--root', type=str, default='/home/pumetu/Purdue/LaneDetection/BDD100k/', help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    return parser.parse_args()

def main():
    args = parse_arg()

    #Load model
    model = YoloMulti().to(device)

    #Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Can try 6e-4
    
    #loss_fn = MultiLoss()
    loss_fn = SegmentationLoss()

    transform = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    #Load BDD100k Dataset
    train_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=True, transform=transform, anchors=ANCHORS)
    val_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=False, transform=transform, anchors=ANCHORS)

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=False)
                  
    imgs, det, lane, drivable = next(iter(train_loader)) # First batch
    model.train()

    for epoch in tqdm.tqdm(range(args.epoch)):
        #--------------------------------------------------------------------------------------
        #Train
        
        #for imgs, det, lane, drivable in train_loader:
        imgs, lane, drivable = imgs.to(device), lane.to(device), drivable.to(device)  
        model.train()
        running_loss = 0

        pdet, pseg = model(imgs)
        lane = lane.squeeze(dim=2)
        loss = loss_fn(pseg, torch.stack(lane, drivable, axis=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        writer.add_scalar("Loss/train", running_loss, epoch)

        writer.flush()
    

if __name__ == '__main__':
    main()
