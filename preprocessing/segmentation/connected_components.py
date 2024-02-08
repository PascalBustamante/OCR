# import the necessary packages
import cv2
import cv2 as cv
import numpy as np
 
def segment(thresh, kernel=None, ):
	if kernel == None:
		kernel = np.ones((3,5), np.uint8)
	
	# Perform dilation and erosion 

	# Apply connected component analysis
	output = cv.connectedComponentsWithStats(
		thresh, 8, cv.CV_32S)
	(numLabels, labels, stats, centroids) = output
	
	# The first component is usually
	# the background and can safely skip it
	for i in range(1, numLabels):
		# Area con the component
		area = stats[i, cv2.CC_STAT_AREA]

		# Filter based on size




# load the input image from disk, convert it to grayscale, and
# threshold it

image = cv2.imread(r"C:\Users\pasca\Data Science\Math Notes Model\OCR\test_data\math_example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

kernel = np.ones((3,3), np.uint8)
lines = cv2.erode(thresh, kernel, iterations=1)
lines = cv2.dilate(lines, kernel, iterations=5)
lines = cv2.GaussianBlur(lines, (5,5), 0)

# apply connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(
	lines, 8, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

# loop over the number of unique connected component labels
for i in range(0, numLabels):
	# if this is the first component then we examine the
	# *background* (typically we would just ignore this
	# component in our loop)
	if i == 0:
		text = "examining component {}/{} (background)".format(
			i + 1, numLabels)
		
	# otherwise, we are examining an actual connected component
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)
		
	# print a status message update for the current connected
	# component
	print("[INFO] {}".format(text))
	
	# extract the connected component statistics and centroid for
	# the current label
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
	(cX, cY) = centroids[i]
	
    # clone our original image (so we can draw on it) and then draw
	# a bounding box surrounding the connected component along with
	# a circle corresponding to the centroid
	output = image.copy()
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
	
	# construct a mask for the current connected component by
	# finding a pixels in the labels array that have the current
	# connected component ID
	componentMask = (labels == i).astype("uint8") * 255
	
	# show our output image and connected component mask
	cv2.imshow("Output", lines)
	cv2.imshow("Connected Component", componentMask)
	cv2.waitKey(0)