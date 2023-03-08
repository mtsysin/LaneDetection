import torch
from torch import nn
from utils import DetectionUtils
from torchvision.utils import draw_bounding_boxes

'''
Author: Pume Tuchinda
'''

def draw_bbox(img, dets):

    def unnorm_bbox(bbox, width, height, i, j, sx, sy):
        '''
        Unormalize the bounding box predictions based on the yolo predictions
        '''
        bbox[:,0] = width * (bbox[:,0] + j) / sx
        bbox[:,1] = height * (bbox[:,1] + i) / sy
        bbox[:,2] = width * (bbox[:,2] / sx)
        bbox[:,3] = height * (bbox[:,3] / sy)

        return bbox

    class_dict = {
        'pedestrian' : 1,
        'rider' : 2,
        'car' : 3,
        'truck' : 4, 
        'bus' : 5, 
        'train' : 6, 
        'motorcycle' : 7,
        'bicycle' : 8,
        'traffic light' : 9,
        'traffic sign' : 10,
        'other vehicle': 11,
        'trailer': 12,
        'other person': 13,
    }
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

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def nms(bboxes, iou_threshold, conf_threshold):
    bboxes = [box for box in bboxes if box[1] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format='midpoint'
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms  

def process_prediction(prediction, img_height, img_width, anchor, C=13):
    batch_size, n_anchors, gy, gx, n_outputs = prediction.shape
    prediction = prediction[0].clone().detach().cpu()
    gridy, gridx = torch.meshgrid([torch.arange(gy), torch.arange(gx)], indexing='ij')
    prediction[..., C+1:C+3] = prediction[..., C+1:C+3].sigmoid()
    prediction[..., C+3:C+5] = prediction[..., C+3:C+5].exp() * anchor

    confidence = prediction[..., C].sigmoid()

    predicted_class = torch.argmax(prediction[..., :C], dim=-1)
    x = (prediction[..., C+1] + gridx) / gx * img_width
    y = (prediction[..., C+2] + gridy) / gy * img_height
    width = prediction[..., C+3] / gx * img_width
    height = prediction[..., C+4] / gy * img_height
    detection = torch.stack((predicted_class, confidence, x, y, width, height), dim=-1).view(batch_size, -1, 6)

    return detection   
    
def get_bboxes(predictions, iou_threshold, conf_threshold):
    batch_size = predictions[0].shape[0]
    for i, prediction in enumerate(predictions):
        anchor = torch.tensor(ANCHORS[i]).view(1, 3, 1, 1, 2)
        detection = process_prediction(prediction, 384, 640, anchor, C=13)
    print(detection.shape)
    nms_b = nms(detection[0], iou_threshold, conf_threshold)

    return nms_b
