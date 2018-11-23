"""

"""

from matplotlib import pyplot as plt
# import numpy as np
import cv2
from histogram import gradient, magnitude_orientation, hog, visualise_histogram
from scipy.ndimage.interpolation import zoom

# load image from filesystem and resize it (reduce computation time)
# img = cv2.imread('/home/chrissi/Bilder/Handy Sicherung 02-17/18-02/DSC_0243.JPG', 0)
img = cv2.imread('/home/chrissi/Bilder/Handy Sicherung 02-17/DCIM/100ANDRO/DSC_0231.JPG', 0)
img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)


gx, gy = gradient(img, same_size=False)
mag, ori = magnitude_orientation(gx, gy)

# show gradient and magnitude
plt.figure(figsize=(15, 10))
plt.subplot(131)
plt.title('original - grey')
plt.imshow(img, cmap=plt.cm.Greys_r)
plt.subplot(132)
plt.title('gradient - gx')
plt.imshow(gx, cmap=plt.cm.Greys_r)
plt.subplot(133)
plt.title('magnitude')
plt.imshow(mag, cmap=plt.cm.Greys_r)

# show orientation deducted from gradient
plt.figure(figsize=(15, 10))
plt.title('orientation')
plt.imshow(ori)
plt.pcolor(ori)
plt.colorbar()

# plot histogram
# make the image bigger to compute the histogram
im1 = zoom(img, 1.5)
h = hog(im1, cell_size=(2, 2), cells_per_block=(1, 1), visualise=False, nbins=9, signed_orientation=False, normalise=True)
im2 = visualise_histogram(h, 8, 8, False)

plt.figure(figsize=(15, 10))
plt.title('HOG features')
plt.imshow(im2, cmap=plt.cm.Greys_r)

plt.show()
