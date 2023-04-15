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
from model.DumbNet import DumbNet
from bdd100k import BDD100k, ANCHORS, BDD_100K_ROOT
#from utils import non_max_supression, mean_average_precission, intersection_over_union
from loss import MultiLoss, SegmentationLoss, DetectionLoss
from utils import Reduce_255

import matplotlib.pyplot as plt
import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/prototype_lane')


torch.cuda.empty_cache()
device = torch.device('cuda')
print(torch.cuda.is_available()) 
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

'''
Author: Pume Tuchinda
'''

LOSS_COUNT = 1
ROOT = "."
USE_DDP = False             # deosn't work currently
USE_PARALLEL = False

INPUT_IMG_TRANSFORM = transforms.Compose([
        transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
        Reduce_255(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

SHUFFLE_OFF = True

gpu_id = None

MODEL = YoloMulti

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
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--root', type=str, default=BDD_100K_ROOT, help='root directory for both image and labels')
    parser.add_argument('--cp', '-checkpoint', type=str, default='', help='path to checpoint of pretrained model')
    parser.add_argument('--sched_points', nargs='+', type=int, default=[250, 400, 1000, 2000], help='sheduler milestones list')
    parser.add_argument('--sched_gamma', type=int, default=0.1, help='gamma for learning rate scheduler')
    parser.add_argument('--save', type=bool, default=True, help='save model flag')
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

    if USE_DDP:
        pass

    #Load model
    if USE_DDP:
        model = MODEL()
        model = DDP(model, device_ids=[gpu_id])
    elif USE_PARALLEL:
        model = MODEL()
        model= torch.nn.DataParallel(model)
        model.to(device)
    else:
        model = MODEL().to(device)
        print(device)

    # model.apply(weights_init)
    model.train()

    #Set optimizer, loss function, and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4, 
        betas=(0.9, 0.99)
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=0.001,   
    # )

    SCHED_STEP = "no"
    # SCHED_STEP = "batch"
    # SCHED_STEP = "epoch"
    if SCHED_STEP != "no":
        "Scheduler is ON"
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01, cycle_momentum=False, step_size_up=100)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sched_points, gamma=args.sched_gamma)
    # loss_fn = MultiLoss()
    loss_fn = DetectionLoss()

    transform = INPUT_IMG_TRANSFORM

    #Load BDD100k Dataset
    train_dataset = BDD100k(root=BDD_100K_ROOT, train=True, transform=transform, anchors=ANCHORS)
    indices = [i for i in range (1)]
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
            det = [d.to(device) for d in det]

            det_pred, _ = model(imgs)

            # print(":::::", next(model.parameters()).device)
            # print("::", device)

            loss = loss_fn(det_pred, det)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # writer.add_scalar("Loss/train", running_loss, epoch)

            if (i+1) % LOSS_COUNT == 0:
                file_log.set_description_str(
                    "[epoch: %d, batch: %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / LOSS_COUNT)
                )
                losses.append(running_loss / LOSS_COUNT)
                running_loss = 0.0

            inner_tqdm.update(1)

            if SCHED_STEP == "batch":
                scheduler.step()

        outer_tqdm.update(1)
        # writer.flush()

        if SCHED_STEP == "epoch":
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
