from matplotlib import pyplot as plt
# import numpy as np
import cv2
from histogram import gradient, magnitude_orientation, hog, visualise_histogram
from scipy.ndimage.interpolation import zoom
import dlib
from skimage.draw import line

detector = dlib.get_frontal_face_detector()


def scale_faces(face_rects, down_scale=1.5):
    faces = []
    for face in face_rects:
        scaled_face = dlib.rectangle(int(face.left() * down_scale),
                                    int(face.top() * down_scale),
                                    int(face.right() * down_scale),
                                    int(face.bottom() * down_scale),)
        faces.append(scaled_face)
    return faces


def detect_faces(image, down_scale=1.5):
    image_scaled = cv2.resize(image, None, fx=1.0/down_scale, fy=1.0/down_scale,
    interpolation=cv2.INTER_LINEAR)
    faces = detector(image_scaled, 0)
    print("Scaled: {}".format(image_scaled.shape))
    return faces


# img = cv2.imread('/home/chrissi/Bilder/Handy Sicherung 02-17/18-02/DSC_0243.JPG', 0)
img = cv2.imread('/home/chrissi/Bilder/Handy Sicherung 02-17/DCIM/100ANDRO/DSC_0231.JPG', 0)
img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

gx, gy = gradient(img, same_size=False)
mag, ori = magnitude_orientation(gx, gy)

faces = detect_faces(img, down_scale=0.5)



# plot histogram
# make the image bigger to compute the histogram
im1 = zoom(img, 1.5)
h = hog(im1, cell_size=(2, 2), cells_per_block=(1, 1), visualise=False, nbins=9, signed_orientation=False, normalise=True)
hog_img = visualise_histogram(h, 8, 8, False)
print(type(hog_img))

dets = detector(img, 1)
print("hog scale {}".format(hog_img.shape))
# add rects on hog image
for face in faces:
    sc = 2
    x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
    print("x, y, w, h: {} {} {} {}".format(x, y, w, h))
    x = int(x*sc)
    y = int(y*sc)
    w = int(w*sc)
    h = int(h*sc)
    print("x, y, w, h scaled: {} {} {} {}".format(x, y, w, h))

    x2, y2 = x, y+h
    x3, y3 = x+w, y+h
    x4, y4 = x+h, y
    print("x2, y2 {}, {}".format(x2, y2))
    print("x3, y3 {}, {}".format(x3, y3))
    print("x4, y4 {}, {}".format(x4, y4))

    rr, cc = line(x, y, w, y)
    # print("rr {}".format(rr.shape))
    # print(type(rr))
    print("cc {}".format(cc.shape))
    # print(type(cc))
    hog_img[rr, cc] = 1
    rr, cc = line(w, y, w, h)
    hog_img[rr, cc] = 1
    rr, cc = line(w, h, x, h)
    hog_img[rr, cc] = 1
    rr, cc = line(x, h, x, y)
    hog_img[rr, cc] = 1
    #cv2.rectangle(hog_img, (x, y), (w, h), (255, 200, 150), 2, cv2.LINE_AA)
    cv2.rectangle(ori, (x, y), (w, h), (255, 200, 150), 2, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (w, h), (255, 200, 150), 2, cv2.LINE_AA)

plt.figure(figsize=(10, 10))
plt.title('HOG features')
plt.imshow(hog_img, cmap=plt.cm.Greys_r)

# show gradient and magnitude
plt.figure(figsize=(15, 10))
plt.title('gradient and magnitude')
plt.subplot(131)
plt.imshow(img, cmap=plt.cm.Greys_r)
plt.subplot(132)
plt.imshow(gx, cmap=plt.cm.Greys_r)
plt.subplot(133)
plt.imshow(mag, cmap=plt.cm.Greys_r)

# show orientation deducted from gradient
plt.figure(figsize=(15, 10))
plt.title('orientation')
plt.imshow(ori)
plt.pcolor(ori)
plt.colorbar()

plt.show()
# subprocess.Popen('deactivate')
