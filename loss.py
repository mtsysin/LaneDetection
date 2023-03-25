'''
YOLOv4 Loss Implementation
- Author: William Stevens
Computes loss given a detected bounding box and the groundtruth data.
As the YOLOv4 paper (https://arxiv.org/abs/2004.10934) indicates, Complete-IOU loss and Focal loss remain to be
effective loss functions. (https://arxiv.org/pdf/1911.08287.pdf, https://arxiv.org/pdf/1708.02002.pdf)
'''

''' IMPORTS '''
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import bbox_iou, xywh_to_xyxy
import math

'''
Complete-IoU Loss: Calculates the detection loss of a bounding box prediction by evaluating three metrics:
Loss related to the IOU, the center coordinate placement, and aspect ratio consistency of detected bounding box
'''
class CompleteIoULoss(nn.Module):    
    def __init__(self, eps:float = 1e-6):
        '''
        Initialize epsilon constant as 10^-6
        '''
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target:torch.Tensor):
        '''
        Params:
        - pred (Tensor): detected bounding box tensor from the model, shape: (N, 4)
        - target (Tensor): bounding box groundtruth data, shape: (M, 4)
        '''
        # Compute IOU using function in utils/utils.py
        iou = bbox_iou(pred, target)

        # Compute penalty term for distance between bounding box centers
        
        # Compute square of distance between bounding box centers
        center_dist_sq = (target[0] - pred[0])**2 + (target[1] - pred[1])**2
        # Compute convex box width and height
        pred_ = xywh_to_xyxy(pred)
        targ_ = xywh_to_xyxy(target)
        convex_width = max(pred_[2], targ_[2]) - min(pred_[0], targ_[0])
        convex_height = max(pred_[3], targ_[3]) - min(pred_[1], targ_[1])
        # Compute square of convex diagonal
        convex_diag_sq = convex_width ** 2 + convex_height ** 2 + self.eps

        # Penalty term is: (center distance)^2 / (convex diagonal length)^2
        dist_penalty = center_dist_sq / convex_diag_sq

        # Compute consistenty of aspect ratio: v
        rho = ((target[:,0] + target[:,2] - pred[0] - pred[2]) ** 2 + (target[:,1] + target[:,3] - pred[1] - pred[3]) ** 2) / 4
        v = (4 / math.pi ** 2) * torch.pow(torch.atan())
        # Compute positive trade-off parameter alpha
        alpha = v / ((1 + self.eps) - iou + v)

        # Return computed Complete-IOU loss
        ciou = iou + dist_penalty - (rho / convex_diag_sq + v * alpha) 
        return (1 - ciou).mean()

'''
Focal Loss: Enhancement to cross entropy loss which improves classification accuracy caused by class imbalances.
''' 
class FocalLoss(nn.Module):
    def __init__(self,
            alpha=None, 
            gamma=0):
        
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        '''
        Params:
        - pred (Tensor): prediction tensor containing confidence scores for each class.
        - target (Tensor): ground truth containing correct class labels.
        - alpha: class weights to represent the class imbalance.
        - gamma: Focal term. Constant, tunable exponent applied to the modulating factor which amplifies
        loss emphasis on difficult learning tasks that result in misclassification.
        '''
        nll_loss = nn.NLLLoss(weight=self.alpha, reduction='none')

        # Weighted cross entropy: alpha * -log(pt)
        log_p = F.log_softmax(pred, dim=-1)
        ce = nll_loss(log_p, target)

        # Get class column from rows
        rows = torch.arange(len(pred))
        log_pt = log_p[rows, target]

        # Focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**(self.gamma)

        focal_loss = focal_term * ce

        return focal_loss.mean()
    
'''
YOLOv4 loss combined for object detection
''' 

class DetLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, target):
        '''
        Params:
        - pred (Tensor): prediction tensor containing confidence scores for each class.
        - target (Tensor): ground truth containing correct class labels.
        - alpha: class weights to represent the class imbalance.
        - gamma: Focal term. Constant, tunable exponent applied to the modulating factor which amplifies
        loss emphasis on difficult learning tasks that result in misclassification.
        '''