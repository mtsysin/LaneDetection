'''
CutMix Data Augmentation  Implementation
- Author: William Stevens
Inspired by the official CutMix paper (https://arxiv.org/pdf/1905.04899.pdf)
'''

import torch
import numpy as np
from torch.distributions.beta import Beta
import torch.nn.functional as F

class CutMix():
    def __init__(self, image_width, image_height, num_classes, alpha=1):
        '''
        Initialize CutMix object
        Params:
        - image_width (int): width of training images
        - image_height (int): height of training images
        - num_classes (int): number of classes in training dataset
        - alpha (float): the "combination ratio", as defined in CutMix paper (default to 1)

        Initialize instance variables
        Define hadamard function call as element-wise multiplication
        '''
        self.image_width = image_width
        self.image_height = image_height
        self.num_classes = num_classes
        self.alpha = alpha


    def get_cutmix(self, im1, im2, lab1, lab2):
        '''
        Obtain CutMix result of two input training images and their labels.
        Params:
        - im1 (Tensor): first training image, shape W x H x C
        - im2 (Tensor): second training image, shape W x H x C
        - lab1 (Tensor): label for first image, shape W x H x C
        - lab2 (Tensor): label for second training image, shape W x H x C

        Returns:
        - im (Tensor): new training image after CutMix, shape W x H x C
        - lab (Tensor): label for new training image after CutMix, shape W x H x C
        '''

        # Resize the second image to match the first image size
        im2_resized = F.interpolate(im2.unsqueeze(0), size=(self.image_height, self.image_width), mode='bilinear', align_corners=False).squeeze(0)

        # Sample the combination ratio lambda from Beta distribution
        lam = Beta(self.alpha, self.alpha).sample().item()

        # Calculate the bounding box size and position for cutmix
        bb_width = int(torch.sqrt(1 - torch.tensor(lam)) * self.image_width)
        bb_height = int(torch.sqrt(1 - torch.tensor(lam)) * self.image_height)
        bb_x = torch.randint(0, self.image_width - bb_width, (1,)).item()
        bb_y = torch.randint(0, self.image_height - bb_height, (1,)).item()

        # Create the mask for cutmix
        mask = torch.zeros_like(im1)
        mask[:, bb_y:bb_y + bb_height, bb_x:bb_x + bb_width] = 1

        # Perform cutmix
        im = im1 * (1 - mask) + im2_resized * mask

        # Create the new label after cutmix
        lab = lab1 * torch.tensor(lam) + lab2 * torch.tensor(1 - lam)

        return im, lab


    
