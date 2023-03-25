'''
BDD100k Dataset Loader
Inspired by Pume Tuchinda's implementation from Lane Detection Fall 2022
'''

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt

class BDD100k(data.Dataset):
    '''
    BDD100k Class object to be used statically
    - Loads the BDD100k .json data files to store groundtruth data in tensors to be sent to model for training
    '''
    def __init__(self, 
            root: str = '../bdd100k/', 
            transforms: transforms = None, 
            training: bool = False, 
            input_size: tuple = (416, 416), 
            anchors: List = [
                [[12, 16], [19, 36], [40, 28]],
                [[36, 75], [76, 55], [72, 146]],
                [[142, 110], [192, 243], [459, 401]]
            ]
        ):
        '''
        Params:
        - root: path to root folder of dataset
        - transforms: data transformations to use
        - training: boolean variable to indicate whether this dataset loader will be used for training or validation
        - input_size: shape of input images. Set to 416x416 for BDD100k images
        - anchors: list of anchor shapes for model. 3 anchors per scale, 9 in total
        '''    
        super().__init__()
        self.root = root
        self.training = training
        self.transforms = transforms
        self.anchors = torch.tensor(anchors)
        self.scale = len(anchors)
        self.input_size = input_size
        self.image_path = self.root + 'images/100k/train/' if self.training else self.root + 'images/100k/val/'
        self.det_path = self.root + 'labels/det_20/det_train.json' if self.training else self.root + 'labels/det_20/det_val.json'
        self.lane_path = self.root + 'labels/lane/masks/train/' if self.training else self.root + 'labels/lane/masks/val/'
        self.drivable_path = self.root + 'labels/drivable/masks/train/' if self.training else self.root + 'labels/drivable/masks/val/'
        
        detections = pd.read_json(self.det_path)
        attributes = pd.DataFrame.from_records(detections.attributes)
        self.detections = pd.concat([detections.drop(labels='attributes', axis=1), attributes], axis=1)
        self.detections.dropna(axis=0, inplace=True)

        self.classes = {
            "pedestrian":1,
            "rider":2,
            "car":3,
            "truck":4,
            "bus":5,
            "train":6,
            "motorcycle":7,
            "bicycle":8,
            "traffic light":9,
            "traffic sign": 10,
            "other vehicle": 11,
            "other person": 12,
            "trailer": 13
        }
        
        self.num_outputs = len(self.classes) + 5
        self.num_classes = len(self.classes)

    def __len__(self):
        '''
        Return length of detections
        '''
        return len(self.detections)

    def __getitem__(self, index):
        '''
        Iterate through .json file at given index to compute detection and segmentation labels
        from original image
        '''
        #Retrieve image from dataset folder
        target = self.detections.iloc[index]
        image = torchvision.io.read_image(self.image_path + target['name'])
        annotations = target['labels']

        #Retrieve bounding box coordinate groundtruth data for all objects in the image
        labels = []
        for object in annotations:
            label = list(object['box2d'].values())
            label.append(self.classes[object['category']])
            labels.append(label)

        #Retrieve lane and drivable area groundtruth data
        lane_path = self.lane_path + target['name'].replace('.jpg', '.png')
        drivable_path = self.drivable_path + target['name'].replace('.jpg', '.png')

        #Call supplementary functions to build detection targets and segmentation targets
        target = {}
        target['detections'] = torch.as_tensor(labels, dtype=torch.float32)
        target['segmentations'] = self._build_seg_target(lane_path, drivable_path)

        #If any data transforms were instructed, apply them
        if self.transforms:
            image = self.transforms(image)

        return image, target
       
    def _build_seg_target(self, lane_path, drivable_path):
        '''
        Build groundtruth data for segmentation
        - Combines the lanes and drivable masks into one 
        Params:
        - lane_path: file path to lane binary mask
        - drivable_path: file path to drivable binary mask
        '''

        #Read binary images of lane mask and drivable area mask into matrices
        lane = plt.imread(lane_path)[..., 0]
        drivable = plt.imread(drivable_path)[..., 0]

        # Use bitwise and with 7 to retrieve lowest 3 bits to correspond to the 9 lane and 3 drivable area types
        # Lane types: road curb, crosswalk, double white, double yellow, double other color, single white, single yellow, single
        # other color, and background
        # Drivable area types: directly drivable area, alternatively drivable area, and background
        lanes = np.bitwise_and(lane, 0b111)
        lane_mask, drivable_mask = [], []

        # Iterate over the 9 lane types and translate data to lane mask 
        for i in range(9):
            lane_mask.append(np.where(lanes==i, 1, 0))
            # Iterate over the 3 drivable area types and translate data to drivable mask 
            if i in range(3):
                drivable_mask.append(np.where(drivable==i, 1, 0))
        # Stack masks together in one matrix, then concatenate
        lane_mask, drivable_mask = np.stack(lane_mask), np.stack(drivable_mask)
        mask = np.concatenate((lane_mask, drivable_mask), axis=0)
        
        return torch.tensor(mask)
    
#Collate batch of images and their targets into a tensor using torch.stack
def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), targets

if __name__=="__main__":
    '''
    Test dataloader func
    '''
    ROOT = "bdd100k/"

    dataset = BDD100k(
        root=ROOT,
        training=True,
    )

    img, target = dataset[4]

    print(target)
    print(img)
    