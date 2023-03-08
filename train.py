import argparse

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from model import YoloMulti
from data_loader import BDD100k
#rom utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss

import matplotlib.pyplot as plt
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/prototype_lane');

device = torch.device('cuda')
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

'''
Author: Pume Tuchinda
'''


ANCHORS = [[(3,9),(5,11),(4,20)], [(7,18),(6,39),(12,31)], [(19,50),(38,81),(68,157)]]

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--root', type=str, default='/home/pumetu/Purdue/LaneDetection/BDD100k/', help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    return parser.parse_args()

def main():
    args = parse_arg()

    #Load model
    model = YoloMulti().to(device)

    #Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = MultiLoss()

    transform = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    #Load BDD100k Dataset
    train_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=True, transform=transform, anchors=ANCHORS)
    val_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=False, transform=transform, anchors=ANCHORS)

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=False)
                  
    for epoch in tqdm.tqdm(range(300)):
        #--------------------------------------------------------------------------------------
        #Train
        for imgs, det, lane, drivable in train_loader:
            imgs, lane, drivable = imgs.to(device), lane.to(device), drivable.to(device)  
            model.train()
            running_loss = 0

            pdet, plane, pdrive = model(imgs)
            lane = lane.squeeze(dim=2)
            loss = loss_fn(pdet, plane, pdrive, det, lane, drivable)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            writer.add_scalar("Loss/train", running_loss, epoch)

        writer.flush()
    

if __name__ == '__main__':
    main()