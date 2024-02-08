import cv2 
import numpy as np
from scipy.ndimage import interpolation as inter
from PIL import Image as im

class ImageProcessor:

    # need to add image resizing 

    # Implement denoiser model here.
    
    def get_skew(bin_src):
        """
        Basic contour deskewing algo.
        The main banefit of this method is that
        it can be applied to ROIs later
        Input:
            bin_src: binary image

        Output:
            deskewed: binary image
        """
        new_src = bin_src.copy()
        blur = cv2.GaussianBlur(new_src, (7,7), 0)

        # Blur along the lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,2))
        dilate = cv2.dilate(blur, kernel=kernel, iterations=2)
        cv2.imshow("thinned", dilate)
        cv2.waitKey()

        # Find cotours
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find largest area contour and surround its min area
        largest = contours[0]
        minAreaRect = cv2.minAreaRect(largest)
        print (minAreaRect)
        # Determine the angle.
        angle = minAreaRect[-1]

        if angle < -45:
            angle += 90

        return -1.0*angle  

    def rotate_image(bin_src, angle):
        new_src = bin_src.copy()
        (h, w) = new_src.shape[:2]
        center = (w // 2, h // 2)  
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_src = cv2.warpAffine(new_src, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return new_src
    
    def adaptive_mean_threshing(src):
        """
        Using opencv's implementation of ADAPTIVE_THRESH_MEAN_C
        input: 
            src: image (ndarray)
        
        output:
            binary_image: ndarray
        """
        if len(src.shape) == 3:
            grey = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        
        elif len(src.shape) == 2:
            grey = src
        
        else:
            raise "Error in input src"
        
        kernel = (3, 1)
        blurred = cv2.GaussianBlur(grey, kernel, 0)

        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       21, 10)
        
        return thresh
    
    def thinning(binary_src):
        """
        Skeletonize the image into 1 pixel wide characters
        Input:
            binary_src: threshed binary image
        
        Output:
            thin_image: binary image
        """
        thinned = cv2.ximgproc.thinning(binary_src, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        # Tesseract prefers BINARY over BINARY_INV since 3.5
        thinned = 255 - thinned
        return thinned
        

path = r"C:\Users\pasca\Data Science\Math Notes Model\OCR\test_data\img_test.png"  ## For testing
src = cv2.imread(path)
thinned = ImageProcessor.adaptive_mean_threshing(src)

angle = ImageProcessor.get_skew(thinned)
print(angle)
deskew = ImageProcessor.rotate_image(thinned,  1*angle)

#deskew = ImageProcessor.deskew(thinned)
thinned = ImageProcessor.thinning(thinned)
cv2.imshow("thinned", deskew)
cv2.waitKey()