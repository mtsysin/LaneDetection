import torch
import numpy as np

class DetectionUtils:
    '''
    Author: Pume Tuchinda
    '''
    def xyxy_to_xywh(self, bbox):
        '''
        Converts bounding box of format x1, y1, x2, y2 to x, y, w, h
        Args:
            bbox: bounding box with format x1, y1, x2, y2
        Return:
            bbox_: bounding box with format x, y, w, h if norm is False else the coordinates are normalized to the height and width of the image
        '''
        bbox_ = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
        bbox_[0] = (bbox[0] + bbox[2]) / 2
        bbox_[1] = (bbox[1] + bbox[3]) / 2
        bbox_[2] = bbox[2] - bbox[0]
        bbox_[3] = bbox[3] - bbox[1]

        return bbox_

    def xywh_to_xyxy(self, bbox):
        '''
        Converts bounding box of format x, y, w, h to x1, y1, x2, y2
        Args:
            bbox: bounding box with format x, y, w, h
        Return:
            bbox_: bounding box with format x1, y2, x2, y2
        '''
        bbox_ = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
        bbox_[:, 0] = (bbox[:, 0] - bbox[:, 2] / 2) 
        bbox_[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2)
        bbox_[:, 2] = (bbox[:, 0] + bbox[:, 2] / 2) 
        bbox_[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2)

        return bbox_
    
    
class SegmentationMetric:
    IGNORE = -1

    def __init__(self):
        self.epsilon = 1e-6
    
    def iou(self, pred, target):
        if (torch.max(target) == 0):
            return self.IGNORE
        intersection = torch.logical_and(pred, target).to(torch.float32).sum(dim=(0,1))
        union = torch.logical_or(pred, target).to(torch.float32).sum(dim=(0,1))
        iou = intersection / (union + self.epsilon)

        return iou
    
    def mean_iou(self, seg, pseg, batch_size):
        seg[seg<=0] = 0
        seg[seg>0] = 255
        pseg[pseg<=0] = 0
        pseg[pseg>0] = 255

        mean_iou = [0] * 12
        iou_counts = [0] * 12
        for class_num in range(12):
            for image_num in range(batch_size):
                iou = self.iou(pseg[image_num][class_num], seg[image_num][class_num])
                if (iou != self.IGNORE):
                    mean_iou[class_num] += float(iou.double())
                    iou_counts[class_num] += 1

        return np.divide(mean_iou, iou_counts, where=iou_counts!=0)