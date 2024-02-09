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

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, C, H, W) layout.
    - x, y: the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    B, C, H, W = img.size()

    # rescale x and y to [0, W-1/H-1]
    x = 0.5 * ((x + 1.0) * (W - 1))
    y = 0.5 * ((y + 1.0) * (H - 1))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # get pixel value at corner coords
    Ia = get_pixel_values(img, x0, y0)
    Ib = get_pixel_values(img, x0, y1)
    Ic = get_pixel_values(img, x1, y0)
    Id = get_pixel_values(img, x1, y1)

    # calculate deltas
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    # compute output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out

if __name__=="__main__":
    img = cv2.imread()
    bilinear_sampler()