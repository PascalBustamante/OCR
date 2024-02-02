import cv2 as cv 
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class FeatureExtractor:
    def __init__(self) -> None:
        pass

    

    def HOG(self, src):    ## Histogram of Oriented Gradients 
        # Python gradient calculation 

        # Read image
        img = src
        img = np.float32(img) / 255.0
        
        # Calculate cartisian gradient
        gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)

        # Polorize gradients ( in degrees )
        mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

        # Define the number of bins for the histogram
        bins = 9

        # Define the size of the cells
        cell_size = (8, 8)

        # Define the size of the blocks
        block_size = (2, 2)

        # Calculate the number of cells in each block
        block_stride = (1, 1)

        # Compute HOG features
        hog = cv.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                         img.shape[0] // cell_size[0] * cell_size[0]),
                               _blockSize=(block_size[1] * cell_size[1],
                                           block_size[0] * cell_size[0]),
                               _blockStride=(block_stride[1] * cell_size[1],
                                             block_stride[0] * cell_size[0]),
                               _cellSize=(cell_size[1], cell_size[0]),
                               _nbins=bins)

        # Compute the HOG features
        hog_features = hog.compute(img)

        # Reshape the features to be used in a machine learning model
        hog_features = hog_features.flatten()

        # The 'hog_features' now contains the HOG features for the image
        return hog_features