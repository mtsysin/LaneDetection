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

from model.YoloMulti import YoloMulti
from bdd100k import BDD100k, ANCHORS, BDD_100K_ROOT
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss, SegmentationLoss, DetectionLoss

import matplotlib.pyplot as plt
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/prototype_lane')

device = torch.device('cuda')
print(torch.cuda.is_available()) 
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

'''
Author: Pume Tuchinda
'''

LOSS_COUNT = 1
ROOT = "."
USE_DDP = False
USE_PARALLEL = True

SHUFFLE_OFF = True

gpu_id = None

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help='devise -- cuda or cpu')
    parser.add_argument('--batch', type=int, default=24, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--root', type=str, default=BDD_100K_ROOT, help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    parser.add_argument('--sched_points', nargs='+', type=int, default=[250, 400], help='sheduler milestones list')
    parser.add_argument('--sched_gamma', type=int, default=0.1, help='gamma for learning rate scheduler')
    parser.add_argument('--save', type=bool, default=True, help='save model flag')
    return parser.parse_args()

def main():

    args = parse_arg()

    if USE_DDP:
        pass

    #Load model
    if USE_DDP:
        model = YoloMulti()
        model = DDP(model, device_ids=[gpu_id])
    elif USE_PARALLEL:
        model = YoloMulti().to(device)
        model= torch.nn.DataParallel(model)
        model.to(device)
    else:
        model = YoloMulti().to(device)
        print(device)

    # model = DDP(model, )
    # print("start:::::", next(model.parameters()).device)

    model.train()

    #Set optimizer, loss function, and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=2e-4, 
        betas=(0.9, 0.99)
    )    

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sched_points, gamma=args.sched_gamma)
    # loss_fn = MultiLoss()
    loss_fn = DetectionLoss()

    transform = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Load BDD100k Dataset
    train_dataset = BDD100k(root=BDD_100K_ROOT, train=True, transform=transform, anchors=ANCHORS)
    indices = [i for i in range (5000)]
    train_dataset = data.Subset(train_dataset, indices)

    # val_dataset = BDD100k(root='/home/pumetu/Purdue/LaneDetection/BDD100k/', train=False, transform=transform, anchors=ANCHORS)

    train_loader = data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch,
                                num_workers=args.num_workers,
                                shuffle = False if USE_DDP or SHUFFLE_OFF else True, 
                                sampler= DistributedSampler(train_dataset) if USE_DDP else None
    )

    # val_loader = data.DataLoader(dataset=val_dataset, 
    #                             batch_size=args.batch,
    #                             num_workers=args.num_workers,
    #                             shuffle=False)

    epochs = args.epochs

    file_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}')  
    outer_tqdm = tqdm.tqdm(total=epochs, desc='Epochs', position=1)
    inner_tqdm = tqdm.tqdm(total=len(train_loader), desc='Batches', position=0)
    losses = []
    for epoch in tqdm.tqdm(range(epochs)):
        if USE_DDP:
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        #--------------------------------------------------------------------------------------
        #Train
        inner_tqdm.refresh() 
        inner_tqdm.reset()
        for i, (imgs, det, seg) in enumerate(train_loader):

            imgs, seg = imgs.to(device), seg.to(device)         # Select correct device for training

            det_pred, _ = model(imgs)

            # print(":::::", next(model.parameters()).device)
            # print("::", device)

            loss = loss_fn(det_pred, det)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            writer.add_scalar("Loss/train", running_loss, epoch)

            if (i+1) % LOSS_COUNT == 0:
                file_log.set_description_str(
                    "[epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / LOSS_COUNT)
                )
                losses.append(running_loss / LOSS_COUNT)
                running_loss = 0.0

            inner_tqdm.update(1)
            # scheduler.step()            # Use for cyclic scheduler

        outer_tqdm.update(1)
        writer.flush()
        scheduler.step()

    if args.save:
        torch.save(model.state_dict(), ROOT+'/model.pt')

    return losses
    
    # untransform = transforms.Compose([
    #     transforms.Resize((720, 1280), interpolation=transforms.InterpolationMode.NEAREST),
    # ])
    # imgs = untransform(imgs)
    # seg = untransform(seg)
    # pseg = untransform(pseg)
    # torch.save(imgs, 'imgs.pt')
    # torch.save(seg, 'seg.pt')
    # torch.save(pseg, 'pseg.pt')


if __name__ == '__main__':
    print("run")
    losses = main()
    plt.plot(losses, label = "Loss")
    plt.ylabel('Loss')
    plt.xlabel(f'Processed batches * {LOSS_COUNT}')
    plt.legend()
    plt.savefig("./out/loss_trace.png")


'''
1231231231231232 tensor([[15., 16.],
        [30., 32.],
        [60., 64.]])

run
1231231231231232 tensor([[[0.8000, 1.0000],
         [0.6333, 1.1250],
         [0.6667, 0.4375]],

        [[2.4000, 4.6875],
         [2.5333, 1.7188],
         [1.2000, 2.2812]],

        [[9.4667, 6.8750],
         [6.4000, 7.5938],
         [7.6500, 6.2656]]])

ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]
# GRID_SCALES = [(12, 20), (24, 40), (48, 80)]
GRID_SCALES = [(48, 80), (24, 40), (12, 20)]
'''
