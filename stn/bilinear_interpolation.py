import numpy as np
import cv2

def get_pixel_values(img, x, y):
    """
    Get raw pixel values at coord (x,y)
    Input:
        img: CV2.image (B, H, W, C)
        x, y: tensors of shape (B*H*W)
    
    Returns:
        output: Tensor (B, H, W, C)
    """
    shape = img.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    