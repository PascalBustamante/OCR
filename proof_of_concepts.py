# import cv2 as cv
# import matplotlib.pyplot as plt
# from skimage.feature import hog
# from skimage import exposure

## Start of a program
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage import measure
from skimage.feature import hog

# Read in our vehicles
# car_images = glob.glob('*.jpeg')

# Define a function to return HOG features and visualization
# Features will always be the first element of the return
# Image data will be returned as the second element if visualize=True
# Otherwise, there is no second return element


def get_hog_features(
    img, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True
):
    # Call skimage hog() with vis=False, feature_vector=True
    blurred_img = cv2.GaussianBlur(
        img, (5, 5), 0
    )  # Adjust kernel size (e.g., (5, 5)) and sigma as needed

    features, hog_image = hog(
        img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=vis,
        feature_vector=feature_vec,
    )

    if vis:
        return features.reshape(-1, orient), hog_image
    else:
        return features


# Function for HOG-based segmentation
def hog_segmentation(image, orient, pix_per_cell, cell_per_block, threshold):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # print("Shape of gray:", gray.shape)

    # Calculate HOG features
    features, _ = get_hog_features(
        gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False
    )

    # Reshape the flattened HOG features back to the original image shape
    num_cells_x = (gray.shape[1] // pix_per_cell) - 1
    num_cells_y = (gray.shape[0] // pix_per_cell) - 1
    reshaped_features = features.reshape(
        (num_cells_y, num_cells_x, cell_per_block, cell_per_block, orient)
    )
    hog_image = np.sum(reshaped_features, axis=(2, 3))

    print("Shape of HOG image:", hog_image.shape)

    # Apply threshold to HOG features
    binary_mask = hog_image > threshold
    print("Shape of binary_mask:", binary_mask.shape)
    print("Unique values in binary_mask:", np.unique(binary_mask))

    # Label connected components in the binary mask
    labeled_mask, num_labels = measure.label(binary_mask, return_num=True)
    print("Shape of labeled_mask:", labeled_mask.shape)
    print("Unique values in labeled_mask:", np.unique(labeled_mask))

    # Get bounding boxes for each labeled region
    regions = measure.regionprops(labeled_mask)

    return labeled_mask, regions, binary_mask


# Read in the image
image = cv2.imread(r"C:\Users\pasca\Data Science\Math Notes Model\OCR\test_data\math_example.png")
#image = mpimg.imread(
    #r"C:\Users\pasca\Data Science\Math Notes Model\OCR\test_data\math_example.png"
#)
# Set parameters for HOG segmentation
orient = 9
pix_per_cell = 8
cell_per_block = 2
threshold = 0.6  # Adjust this threshold as needed

# Perform HOG-based segmentation
labeled_mask, regions, binary_mask = hog_segmentation(
    image, orient, pix_per_cell, cell_per_block, threshold
)

plt.figure(figsize=(8, 4))
plt.imshow(labeled_mask[:, :, 0], cmap="gray")
plt.title("labeled_mask")
plt.axis("off")
plt.show()

# Visualize the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
ax1.imshow(image, cmap="gray")
ax1.set_title("Original Image")

# Visualize binary mask
ax2.imshow(binary_mask[:, :, 0], cmap="gray")
ax2.set_title("Binary Mask")

# Visualize labeled regions
ax3.imshow(labeled_mask[:, :, 0], cmap="nipy_spectral")
ax3.set_title("Connected Components")

# Overlay labeled regions on the original image
for i in range(2):
    slice_2d = labeled_mask[:, :, i]
    regions = measure.regionprops(slice_2d)
    # Process regions for the current 2D slice
    overlay = image.copy()
    for region in regions:
        # print(region.bbox)
        # minr, minc, maxr, maxc = region.bbox
        # overlay[minr:maxr, minc:maxc, 0] = 1  # Red overlay (set red channel to 1)

        # Extract bounding box in 2D slice for chosen orientation
        minr, minc, maxr, maxc = (
            region.bbox[0],
            region.bbox[1],
            region.bbox[2],
            region.bbox[3],
        )

        # Plot bounding box
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax3.plot(bx, by, "-b", linewidth=2.5)

        # Plot centroid and orientation
        print(region.orientation)
        y0, x0 = region.centroid
        orientation = region.orientation
        print(orientation)
        x1 = x0 + np.cos(orientation) * 0.5 * region.major_axis_length
        y1 = y0 - np.sin(orientation) * 0.5 * region.major_axis_length
        x2 = x0 - np.sin(orientation) * 0.5 * region.minor_axis_length
        y2 = y0 - np.cos(orientation) * 0.5 * region.minor_axis_length

        ax3.plot((x0, x1), (y0, y1), "-r", linewidth=2.5)
        ax3.plot((x0, x2), (y0, y2), "-r", linewidth=2.5)
        ax3.plot(x0, y0, ".g", markersize=15)

# Visualize the overlay on the original image
fig, ax4 = plt.subplots(figsize=(6, 6))
ax4.imshow(overlay)
ax4.set_title("Overlay on Original Image")

plt.show()
