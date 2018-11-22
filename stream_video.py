from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import numpy as np
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
resolution = (640, 480) # (640, 480)
camera.resolution = resolution
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=resolution)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
ii = 0

def detect(image):
	s = time.time()
	winStride = (4, 4) # (4,4)
	padding = (8, 8) #(8,8)
	scale = 1.05 #1.05
	image = imutils.resize(image, width=min(300, image.shape[1]))
	(rects, weights) = hog.detectMultiScale(image, winStride=winStride,
		padding=padding, scale=scale)
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	print(time.time() - s)
	return image


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	ii += 1
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array	
	image = detect(image)
 
	# show the frame...
	# cv2.imwrite("out/Frame" + str(ii) + ".jpg", image)
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
	#if ii == 10:
		break