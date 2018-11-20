import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('/home/chrissi/Bilder/Handy Sicherung 02-17/18-02/DSC_0243.JPG', 0)
img = cv2.imread('/home/chrissi/Bilder/Handy Sicherung 02-17/DCIM/100ANDRO/DSC_0231.JPG', 0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# Find magnitude and angle
magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
angle = np.arctan2(sobely, sobelx) * (180 / np.pi)

dx, dy = np.gradient(img)
# arr = np.array(dx, dy)
# dxy_img = arr.reshape(img.shape)

images = [img, sobelx, sobely, magnitude, angle]
titles = ['img', 'sobelx', 'sobely', 'magnitude', 'angle']

plt.figure(figsize=(10, 10))
for i in range(len(images)):
    #plt.subplot(3, 2, i+1,  autoscale_on=True)
    #plt.imshow(images[i], 'gray')
    #plt.title(titles[i]), plt.xticks([]), plt.yticks([])
    images[i] = cv2.resize(images[i], None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    cv2.imshow(titles[i], images[i])

#plt.show()

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
