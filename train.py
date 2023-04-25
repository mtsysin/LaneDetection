import argparse

import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../yolov4')

from model.model import YoloMulti
from bdd100k import BDD100k
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss, SegmentationLoss, DetectionLoss

import matplotlib.pyplot as plt
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/prototype_lane');
'''
device = torch.cuda.set_device(1)
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
'''
torch.cuda.empty_cache()
device = torch.device('cuda')
print(torch.cuda.is_available()) 
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

LOSS_COUNT = 1
USE_PARALLEL = True
ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers')
    parser.add_argument('--root', type=str, default='/data/stevenwh/bdd100k/', help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    return parser.parse_args()


# Initialize weights:
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 1)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.5)
        torch.nn.init.constant_(m.bias.data, 0)

def main():
    args = parse_arg()

    #Load model
    if USE_PARALLEL:
        model = YoloMulti()
        model= torch.nn.DataParallel(model)
        model.to(device)
    else:
        model = YoloMulti().to(device)

    model.train()

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
    #val_loader = data.DataLoader(dataset=val_dataset, 
    #                            batch_size=args.batch,
    #                            num_workers=args.num_workers,
    #                            shuffle=False)
                  
    imgs, _, seg = next(iter(train_loader)) # First batch
    model.train()
    
    epochs = args.epochs

    file_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}')  
    outer_tqdm = tqdm.tqdm(total=epochs, desc='Epochs', position=1)
    inner_tqdm = tqdm.tqdm(total=len(train_loader), desc='Batches', position=0)
    losses = []
    for epoch in tqdm.tqdm(range(epochs)):
        
        running_loss = 0.0
        #--------------------------------------------------------------------------------------
        #Train
        
        inner_tqdm.refresh() 
        inner_tqdm.reset()
        for i, (imgs, det, seg) in enumerate(train_loader):
            imgs, seg = imgs.to(device), seg.to(device)  

            pdet, pseg = model(imgs)
        
            loss = loss_fn(pseg, seg.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            #writer.add_scalar("Loss/train", running_loss, epoch)

            if (i+1) % LOSS_COUNT == 0:
                file_log.set_description_str(
                    "[epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / LOSS_COUNT)
                )
                losses.append(running_loss / LOSS_COUNT)
                running_loss = 0.0

            inner_tqdm.update(1)
            #writer.flush()
    

    if args.save:
        torch.save(model.state_dict(), '/data/stevenwh/yolov4/out/model.pt')
    '''
    torch.save(imgs, 'out/imgs.pt')
    torch.save(seg, 'out/seg.pt')
    torch.save(pseg, 'out/pseg.pt')
    '''
    return losses


if __name__ == '__main__':
    print("run")
    losses = main()
    plt.plot(losses, label = "Loss")
    plt.ylabel('Loss')
    plt.xlabel(f'Processed batches * {LOSS_COUNT}')
    plt.legend()
    plt.savefig("./out/loss_trace.png")
