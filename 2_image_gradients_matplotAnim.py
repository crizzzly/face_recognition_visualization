import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
import time

ms_last, ms_now = time.time(), time.time()

images = []
titles = []
axes = []

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# FFMpegWriter = animation.writers('ffmpeg')
# metadata = dict(title='gradients from vid', artist='chrissi', comment='different')

fig = plt.figure(figsize=(10, 10))
plt.ion()

while not (cap.isOpened()):
    pass

ret, frame = cap.read()

if ret is True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)

    dx, dy = np.gradient(gray)
    # arr = np.array(dx, dy)
    # dxy_img = arr.reshape(img.shape)
    images = [gray,  magnitude, angle]  # sobelx, sobely,
    titles = ['img', 'magnitude', 'angle']  # 'sobelx', 'sobely',
    for i in range(len(images)):
        ax = fig.add_subplot(3, 2, i+1,  autoscale_on=True)
        ax.imshow(images[i], 'gray')
        axes.append(ax)
plt.show()
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret is True:
        ms_now = time.time()
        if ms_now-ms_last >= 1:
            ms_last = ms_now
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

            # Find magnitude and angle
            magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
            angle = np.arctan2(sobely, sobelx) * (180 / np.pi)

            # dx, dy = np.gradient(gray)

            images = [gray, magnitude, angle]  # sobelx, sobely,
            titles = ['img', 'magnitude', 'angle']  # 'sobelx', 'sobely',

            for i in range(len(images)):
                axes[i].imshow(images[i], 'gray')
            fig.canvas.draw()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

cv2.destroyAllWindows()
