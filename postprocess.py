import torch
from torch import nn
from utils import DetectionUtils
from torchvision.utils import draw_bounding_boxes
from bdd100k import ANCHORS, CLASS_DICT
from evaluate import DetectionMetric
'''
Author: Pume Tuchinda
'''

detection_metric = DetectionMetric()

def draw_bbox(img, dets):
    """
    Draws bboxes on an image. Accepts a single image with NO batch dimension present
    """
    
    def unnorm_bbox(bbox, width, height, i, j, sx, sy):
        '''
        Unormalize the bounding box predictions based on the yolo predictions
        '''
        bbox[:,0] = width * (bbox[:,0] + j) / sx
        bbox[:,1] = height * (bbox[:,1] + i) / sy
        bbox[:,2] = width * (bbox[:,2] / sx)
        bbox[:,3] = height * (bbox[:,3] / sy)

        return bbox

    class_dict = CLASS_DICT
    num_to_class = {i:s for s,i in class_dict.items()}

    utils = DetectionUtils()

    img = img.to(torch.uint8)
    _, height, width = img.shape

    for det in dets:
        scale_pred = det[0]
        sy, sx = scale_pred.shape[1], scale_pred.shape[2]
        for i in range(sy):
            for j in range(sx):
                conf = scale_pred[:, i, j, 13]
                if conf.any() == 1.:
                    scale = torch.argmax(conf)
                    pred = scale_pred[scale]
                    _, pclass = torch.max(pred[..., :13], dim=2)
                    class_pred = num_to_class[pclass[i, j].item()]
                    label = [class_pred]
                    bbox = pred[i, j, 14:19].clone().unsqueeze(0)
                    bbox = unnorm_bbox(bbox, width, height, i ,j, sx, sy)
                    bbox = utils.xywh_to_xyxy(bbox)
                    img = draw_bounding_boxes(img, bbox, width=3, labels=label, colors=(100, 250, 150)) 

    return img

def nms(bboxes_batch, iou_threshold, conf_threshold):
    bboxes_batch_after_nms = []
    for bboxes in bboxes_batch:

        mask = bboxes[:, 1] >= conf_threshold
        bboxes = bboxes[mask]

        # Sort the boxes by their confidence scores in descending order
        confidences = bboxes[:, 1]
        indices = torch.argsort(confidences, descending=True)
        bboxes = bboxes[indices]

        bboxes_after_nms = []

        while len(bboxes) > 0:                                                   # While we still have bboxes
            chosen_box = bboxes[0]

            # bboxes = [
            #     box for box in bboxes
            #     if box[0] != chosen_box[0] or
            #     detection_metric.box_iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), xyxy=False) < iou_threshold
            # ]

            mask = detection_metric.box_iou(chosen_box[None ,2:], bboxes[... ,2:], xyxy=False).squeeze(-1) < iou_threshold
            bboxes = bboxes[mask]

            bboxes_after_nms.append(chosen_box)

        bboxes_batch_after_nms.append(bboxes_after_nms)

    return bboxes_batch_after_nms

def process_prediction(prediction, img_height, img_width, anchor, C=13):
    """
    Takes a batch of predictions and converts them to a tensor of all bounding boxes with shape
    (batch, # of bounding boxes, 6 [class, confidence, x, y, w, h])
    """
    batch_size, n_anchors, gy, gx, n_outputs = prediction.shape                             # Get dimensions of prediction
    # prediction = prediction[0].clone().detach().cpu()                                       # Use predictions for the first batch
    gridy, gridx = torch.meshgrid([torch.arange(gy), torch.arange(gx)], indexing='ij')
    prediction[..., C+1:C+3] = prediction[..., C+1:C+3].sigmoid()                           # Convert predictions to the format where everything is represented as a fraction of a cell size
    prediction[..., C+3:C+5] = prediction[..., C+3:C+5].exp() * anchor

    confidence = prediction[..., C].sigmoid()                                               # clamp the confidence between 0 and 1 so we can represent it as percentage

    predicted_class = torch.argmax(prediction[..., :C], dim=-1)
    x = (prediction[..., C+1] + gridx) / gx * img_width                                     # Convert x and y to acual pixel values on image
    y = (prediction[..., C+2] + gridy) / gy * img_height
    width = prediction[..., C+3] / gx * img_width                                           # Do the same for w and h
    height = prediction[..., C+4] / gy * img_height
    detection = torch.stack((predicted_class, confidence, x, y, width, height), dim=-1).view(batch_size, -1, 6)

    return detection   
    
def get_bboxes(predictions, iou_threshold, conf_threshold):
    """
    Takes predicted values (3 scales with each scale batched)
    Outputs list of lists: for each batch output a list of acceptable bounding boxes
    """
    detections = []
    for i, prediction in enumerate(predictions):                                        # For all prediction scales: i - number of scale, prediction - prediction for the corresponding scale
        anchor = torch.tensor(ANCHORS[i]).view(1, 3, 1, 1, 2)                           # Get the anchor boxes corresponding to the chosen scale, view: (Batch, anchor index, sy, sx, box dimension)
        detections.append(process_prediction(prediction, 384, 640, anchor, C=13))       # Process detection
    detection = torch.cat(tuple(detections), dim=1)
    print(detection.shape)
    print(detection[1][2:7, ...])
    nms_b = nms(detection, iou_threshold, conf_threshold)                            # Perform non-maximum suppression on the resulting list of bounding boxes

    return nms_b

if __name__=="__main__":
    """Try to run this for sample prediction"""
    pred1 = torch.rand(16, 3, 10, 20, 18)
    pred2 = torch.rand(16, 3, 10, 20, 18)
    pred3 = torch.rand(16, 3, 10, 20, 18)
    pred = [pred1, pred2, pred3]

    nms_b = get_bboxes(pred, 0.7, 0.7)