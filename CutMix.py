'''
CutMix Data Augmentation  Implementation
- Author: William Stevens
Inspired by the official CutMix paper (https://arxiv.org/pdf/1905.04899.pdf)
'''

import torch
import numpy as np

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
        self.w = image_width
        self.h = image_height
        self.c = num_classes
        self.a = alpha

    def hadamard(self, m, x):
        # Element-wise product of two tensors
        return m * x

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


        # Initialize m tensor and patch tensor as full of ones, shape W x H
        m = torch.ones((self.w, self.h)).long()  # convert to long type
        patch = torch.ones((self.w, self.h)).long()  # convert to long type

        # Compute lambda from uniform distribution (same as beta distribution Beta(alpha, alpha))
        # Will get value in between 0 and 1
        lam = np.random.beta(self.a, self.a)

        # Compute patch bounding box coordinates: Px, Py, Pw, Ph. From paper:
        # Px: random int from 0 to W
        # Py: random int from 0 to H
        # Pw: W * sqrt(1 - lambda)
        # Ph: H * sqrt(1 - lambda)
        Px = np.random.randint(self.w)
        Py = np.random.randint(self.h)
        Pw = int(self.w * np.sqrt(1 - lam))
        Ph = int(self.h * np.sqrt(1 - lam))

        # Fill patch tensor with zeros outside of the patch bounding box computed above
        # Bitwise XOR with original mask M to get patch mask
        patch[0:self.w,0:Py] = 0
        patch[0:self.w,Py+Ph:self.h] = 0
        patch[0:Px,Py:Py+Ph] = 0
        patch[Px:Px+Pw,Py:Py+Ph] = 0
        m = torch.bitwise_xor(m, patch)

        m = m.expand(-1, -1, im1.size(2))
        # Compute new cutmix training image by using hadamard product of m and im1, plus 1-m hadamard im2
        im = (self.hadamard(m, im1)) + (self.hadamard(1-m, im2))

        # Compute corresponding training label by using lambda * label1 + (1 - lambda) * label2
        lab = lam * lab1 + (1 - lam) * lab2

        return im, lab


    
