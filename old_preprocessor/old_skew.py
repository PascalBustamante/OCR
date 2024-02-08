import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter

img = im.open("test_data\img_test.png")

# Convert to binary
wd, ht = img.size
pix = np.array(img.convert("1").getdata(), np.uint8)
bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
plt.imshow(bin_img, cmap="gray")
plt.savefig("binary.png")


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


delta = 10
limit = 180
angles = np.arange(-limit, limit + delta, delta)
print(angles)
scores = []

for angle in angles:
    hist, score = find_score(bin_img, angle)
    scores.append(score)
    print(hist, score)

best_score = max(scores)
best_angle = angles[scores.index(best_score)]
print("Best Angle: {}".format(best_angle))

# Correct skew
data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
img2 = im.fromarray((255 * data).astype("uint8")).convert("RGB")
img2.save("skew_corrected.png")
