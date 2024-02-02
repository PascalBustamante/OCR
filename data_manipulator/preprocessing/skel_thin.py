import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# def skeletonization(path):
path = "test_data\math_example.png"  ## For testing
src = cv.imread(path)
size = np.size(src)
cv.imshow("original", src)

# wd, ht = src.size
# pix = np.array(src.convert('1').getdata(), np.uint8)
# bin_img = 1 - (pix.reshape((ht, wd))/ 255.0)

skel = np.zeros(src.shape, np.uint8)

# ret, src = cv.threshold(src, 127, 255, cv.THRESH_BINARY_INV)


titles = ["Original Image", "Adaptive Gaussian Thresholding"]
# images = [src, th3]

padded_src = cv.copyMakeBorder(src, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
gray_src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
_, binary_src = cv.threshold(gray_src, 128, 255, cv.THRESH_BINARY_INV)

thinned = cv.ximgproc.thinning(binary_src, thinningType=cv.ximgproc.THINNING_GUOHALL)
thinned = thinned[1:-1, 1:-1]

cv.imshow("thinned", thinned)

# for i in range(2):
#   plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#  plt.title(titles[i])
# plt.xticks([]),plt.yticks([])
# plt.show()


cv.imshow("Skel", skel)
cv.waitKey(0)
cv.destroyAllWindows()
