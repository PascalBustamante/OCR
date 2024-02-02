import cv2 as cv
import numpy as np

from database_manager import DataSample


class Preprocessor:
    def __init__(self, data_sample, batch_size, transforms: []) -> None:
        self.data = data_sample
        self.batch_size = batch_size
        self.transforms = transforms

    def skeletonization(self, data_sample):
        """
        Perform skeletonization on the images in the data sample.
        """
        for idx, filename in enumerate(data_sample.get_filenames()):
            image = data_sample.data_block[idx]
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            _, binary_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY_INV)
            thinned = cv.ximgproc.thinning(
                binary_image, thinningType=cv.ximgproc.THINNING_GUOHALL
            )
            data_sample.data_block[idx] = thinned
