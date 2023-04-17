import argparse

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import sys
from CutMix import CutMix
import concurrent.futures
sys.path.append('../yolov4')

from model.model import YoloMulti
from bdd100k import BDD100k
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss, SegmentationLoss, DetectionLoss

import matplotlib.pyplot as plt
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/prototype_lane')

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
    parser.add_argument('--root', type=str, default='/data/li4583/bdd100k/', help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    return parser.parse_args()

def main(alpha, cutmix_percentage):
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
    train_dataset = BDD100k(root='/data/li4583/bdd100k/', train=True, transform=transform, anchors=ANCHORS)
    val_dataset = BDD100k(root='/data/li4583/bdd100k/', train=False, transform=transform, anchors=ANCHORS)
    cutmix = CutMix(image_width=640, image_height=384, num_classes=len(train_dataset.class_dict), alpha=alpha)

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle=False)
                  
    imgs, det, seg = next(iter(train_loader)) # First batch
    model.train()

    for epoch in tqdm.tqdm(range(args.epoch)):
        #--------------------------------------------------------------------------------------
        #Train
        
        #for imgs, det, lane, drivable in train_loader:
        for imgs, det, seg in train_loader:
                try:
                    # Iterate over all images in the batch
                    for i in range(imgs.size(0)):
                        # Apply CutMix only with the given probability
                        if np.random.rand() < cutmix_percentage:
                            # Get a random index from the current batch, different from the current index
                            index2 = torch.randint(1, imgs.size(0) - 2, (1,))
                            if index2 >= i:
                                index2 += 1
                            index2 = index2.item()
                            # Apply CutMix to the selected images and labels
                            cutmix_img, cutmix_label = cutmix.get_cutmix(imgs[i], imgs[index2], det[i], det[index2])
                            # Replace the original images and labels with the CutMix-augmented versions
                            imgs[i] = cutmix_img
                            det[i] = cutmix_label
                            
                except IndexError as e:
                    print(f"IndexError: {e}. batch size={imgs.size(0)}, index2={index2}, i={i}")
                    raise e

        imgs, seg = imgs.to(device), seg.to(device)  
        model.train()
        running_loss = 0

        pdet, pseg = model(imgs)
        
        loss = loss_fn(pseg, seg.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        writer.add_scalar("Loss/train", running_loss, epoch)

        writer.flush()
    

    torch.save(imgs, 'out/imgs.pt')
    torch.save(seg, 'out/seg.pt')
    torch.save(pseg, 'out/pseg.pt')
    return running_loss, alpha, cutmix


if __name__ == '__main__':
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  
    cutmix_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]  
    best_alpha = None
    best_cutmix_percentage = None
    best_loss = float('inf')
    for alpha in alpha_values:
        for cutmix_percentage in cutmix_percentages:
            loss, alpha, cutmix_percentage = main(alpha, cutmix_percentage)
            if loss < best_loss:
                best_alpha = alpha
                best_cutmix_percentage = cutmix_percentage
                best_loss = loss

    print("Best alpha:", best_alpha)
    print("Best CutMix percentage:", best_cutmix_percentage)

    print("Best alpha:", best_alpha)
    print("Best CutMix percentage:", best_cutmix_percentage)
    
    
    
    
    """    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  
    cutmix_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]  
    best_alpha = None
    best_cutmix_percentage = None
    best_loss = float('inf')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        
        for alpha in alpha_values:
            for cutmix_percentage in cutmix_percentages:
                futures.append(executor.submit(main, alpha, cutmix_percentage))

        for future in concurrent.futures.as_completed(futures):
            loss, alpha, cutmix_percentage = future.result()
            if loss < best_loss:
                best_alpha = alpha
                best_cutmix_percentage = cutmix_percentage
                best_loss = loss

    print("Best alpha:", best_alpha)
    print("Best CutMix percentage:", best_cutmix_percentage)

    print("Best alpha:", best_alpha)
    print("Best CutMix percentage:", best_cutmix_percentage)"""
