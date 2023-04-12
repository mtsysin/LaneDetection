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
    
    def resize_nearest(image, width, height):
        """
        Resizes an image using nearest-neighbor interpolation.
        Args:
            image: The image to be resized using nearest-neighbor interpolation.
            width: The desired width of the resized image.
            height: The desired height of the resized image.
        Returns:
            new_image: A new image that has been resized to the specified width and height using nearest-neighbor interpolation.
        """
        # Create a new blank image of the desired size
        new_image = Image.new("RGB", (width, height))
        
        # Calculate the scaling factors for width and height
        scale_x = image.width / width
        scale_y = image.height / height
        
        # Loop over every pixel in the new image
        for x in range(width):
            for y in range(height):
                # Calculate the corresponding pixel in the original image
                px = int(x * scale_x)
                py = int(y * scale_y)
                
                # Get the RGB color of the pixel in the original image
                color = image.getpixel((px, py))
                
                # Set the pixel color in the new image
                new_image.putpixel((x, y), color)
        
        return new_image