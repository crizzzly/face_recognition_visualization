# face_recognition_visualization
Visualization of single steps used in HOG-based face recognition

trying out different ways to understand and visualize single steps used for hog-based face recognition.

#### 1_image_gradients.py
uses cv2.Sobel to get the directional change of itensity of the image in x and y-direction.   
np.sqrt calculates the magnitude of change per pixel based on sobel_x and sobel_y
np.arctan is used to get direction of highest change per pixel, also based on sobel_x and sobel_y

results of each calculation are shown as images in one frame using matplotlib.subplot

#### 1_image_gradients_cv2vid.py
uses cv2.VideoCapture on webcam to display live results of sobel_x and sobel_y in two seperate frames

#### 2_image_gradients_matplotAnim.py
also uses cv2.VideoCapture to capture live video from webcam
uses matplotlib to display greyscale image, magnitude and angle in single frame

#### 3_image_histogram.py
result of playing around with python-hog found in https://github.com/JeanKossaifi/python-hog.
with this we can display the calculated vectors (direction and magnitude) as image, but the results of diection and magnitude are less detailed than previous results

#### 3_image_histogram-mark_faces.py
still trying to mark the detected faces on the histogram image
