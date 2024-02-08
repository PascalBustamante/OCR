import numpy as np
import torch
import cv2

def get_pixel_values(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, C, H, W)
    - x: flattened tensor of shape (B*H*W)
    - y: flattened tensor of shape (B*H*W)
    Returns
    -------
    - output: tensor of shape (B, C, H, W)
    """
    shape = x.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = torch.arange(0, batch_size).unsqueeze(1).unsqueeze(1)
    batch_idx = batch_idx.expand(batch_idx, height, width)
    
    indices = torch.stack([batch_idx, y, x], dim=3)

    return torch.gather(img, 2, indices)