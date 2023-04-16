import torch
from torch.utils.data import DataLoader
import unittest
import torchvision.transforms as transforms
import torch.utils.data as data
import time
import sys
sys.path.append('../yolov4')

from bdd100k import BDD100k


class TestBDD100k(unittest.TestCase):
    def test_single_process_dataloader(self):
        train_dataset = BDD100k(root='/data/li4583/bdd100k/', train=True)
        self._check_dataloader(train_dataset, num_workers=0)     
        test_dataset = BDD100k(root='/data/li4583/bdd100k/', train=False)
        self._check_dataloader(test_dataset, num_workers=0)

    def test_multi_process_dataloader(self):
        train_dataset = BDD100k(root='/data/li4583/bdd100k/', train=True)
        self._check_dataloader(train_dataset, num_workers=2)
        test_dataset = BDD100k(root='/data/li4583/bdd100k/', train=False)
        self._check_dataloader(test_dataset, num_workers=2)

    def _check_dataloader(self, data, num_workers):
        """This function only tests that the loading process throws no error"""
        print("Initializing data loader...")
        loader = DataLoader(data, batch_size=4, num_workers=num_workers)
        print("Loading images...")
        for _ in loader:
            pass
        print("Success")

    def test_target(self):
         dataset = BDD100k(root='/data/li4583/bdd100k/', train=False)
         loader = DataLoader(dataset, batch_size=4)
         for img, target in loader:
             pass


#Test Dataloader 1
'''
test = TestBDD100k()
test.test_multi_process_dataloader()
'''
#Test Dataloader 2

print("Beginning dataloader tests...")

ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]

transform = transforms.Compose([
transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST),
])
#Load BDD100k Dataset
train_dataset = BDD100k(root='/data/li4583/bdd100k/', train=True, transform=transform, anchors=ANCHORS)
val_dataset = BDD100k(root='/data/li4583/bdd100k/', train=False, transform=transform, anchors=ANCHORS)

train_start = time.time()

train_loader = data.DataLoader(dataset=train_dataset,
                            batch_size=2,
                            num_workers=12, # start at 12, go up to 20 to see which is faster
                            shuffle=True)   # 12 WAS FASTER. And doing 20 gave a warning
train_end = time.time()
print("Training: ", (train_end - train_start))

val_start = time.time()
val_loader = data.DataLoader(dataset=val_dataset, 
                            batch_size=2,
                            num_workers=12,
                            shuffle=False)
val_end = time.time()
print("Validation: ", (val_end - val_start))

imgs, dets, lanes, drives = next(iter(val_loader))

