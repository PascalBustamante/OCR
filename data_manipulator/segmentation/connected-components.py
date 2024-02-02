from scipy.ndimage import label
from scipy.ndimage import binary_closing
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

import matplotlib.pylab as plt
import numpy as np
import cv2

# Read in the image
image = cv2.imread(r"C:\Users\pasca\Data Science\Math Notes Model\STN\test_data\math_example.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve segmentation
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use adaptive thresholding to segment the image
_, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Label connected components in the binary mask
labeled_mask, num_labels = cv2.connectedComponents(binary_mask)

# Get bounding boxes for each labeled region
regions = cv2.boundingRect(binary_mask)

# Visualize the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')

ax2.imshow(binary_mask, cmap='gray')
ax2.set_title('Binary Mask')

ax3.imshow(labeled_mask, cmap='nipy_spectral')
ax3.set_title('Connected Components')

plt.show()