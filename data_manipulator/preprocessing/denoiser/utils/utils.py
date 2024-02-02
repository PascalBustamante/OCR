import cv2 as cv
import numpy as np

def blur_threshold(image, eps=1e-7):
    # apply a median blur to the image and then subtract the blurred
	# image from the original image to approximate the foreground
	blur = cv.medianBlur(image, 5)
	foreground = image.astype("float") - blur
	
	# threshold the foreground image by setting any pixels with a
	# value greater than zero to zero
	foreground[foreground > 0] = 0
	
    # apply min/max scaling to bring the pixel intensities to the
	# range [0, 1]
	minVal = np.min(foreground)
	maxVal = np.max(foreground)
	foreground = (foreground - minVal) / (maxVal - minVal + eps)
	
    # return the foreground-approximated image
	return foreground