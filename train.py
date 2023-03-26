import argparse

import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

<<<<<<< HEAD
from model import YOLOP
from bdd100k import BDD100k
# from utils import non_max_supression, mean_average_precission, intersection_over_union
# from loss import MultiLoss
=======
from model import YoloMulti
from bdd100k import BDD100k
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss

>>>>>>> master
import matplotlib.pyplot as plt
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/prototype_lane');

<<<<<<< HEAD
device = torch.device('cuda')
print(torch.cuda.is_available())
=======
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.cuda.set_device(1)
>>>>>>> master
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

'''
Author: Pume Tuchinda
'''

BDD_100K_ROOT = ".bdd100k/"

ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='devise -- cuda or cpu')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--root', type=str, default=BDD_100K_ROOT, help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    parser.add_argument('--sched_points', nargs='+', type=int, default=[20, 40], help='sheduler milestones list')
    parser.add_argument('--sched_gamma', type=int, default=0.1, help='gamma for learning rate scheduler')
    return parser.parse_args()

def main():
    args = parse_arg()

    #Load model
    model = YOLOP().to(device)

    model.train()

    #Set optimizer, loss function, and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4, 
        betas=(0.9, 0.99)
    )    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)
    loss_fn = MultiLoss()

    transform = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Load BDD100k Dataset
<<<<<<< HEAD
    train_dataset = BDD100k(root=BDD_100K_ROOT, train=True, transform=transform, anchors=ANCHORS)
    # val_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=False, transform=transform, anchors=ANCHORS)
=======
    train_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=True, transform=transform, anchors=ANCHORS)
    val_dataset = BDD100k(root='/data/stevenwh/bdd100k/', train=False, transform=transform, anchors=ANCHORS)
>>>>>>> master

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=True)
    # val_loader = data.DataLoader(dataset=val_dataset, 
    #                             batch_size=args.batch,
    #                             num_workers=args.num_workers,
    #                             shuffle=False)

    epochs = args.epochs

    file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')  
    outer_tqdm = tqdm.tqdm(total=epochs, desc='Epochs', position=0)
    for epoch in tqdm.tqdm(range(epochs)):
        #--------------------------------------------------------------------------------------
        #Train
        inner_tqdm = tqdm.tqdm(total=len(train_loader), desc='Batches', position=0)
        for imgs, det, seg in train_loader:
            imgs, seg = imgs.to(device), seg.to(device)  
            running_loss = 0

<<<<<<< HEAD
            det_pred, seg_pred = model(imgs)
=======
            pdet, pseg = model(imgs)
>>>>>>> master
            lane = lane.squeeze(dim=2)
            loss = loss_fn(pdet, pseg, det, lane, drivable)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            writer.add_scalar("Loss/train", running_loss, epoch)


            inner_tqdm.update(1)
        outer_tqdm.upadte(1)

        writer.flush()
        scheduler.step()

    

if __name__ == '__main__':
    main()
