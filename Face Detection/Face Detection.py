

# importing the necessary packages
from imutils.video import VideoStream
import urllib.request
import numpy as np
import imutils
import time
import cv2
import os
import PIL
from PIL import Image

mywidth = 400



# loading our serialized model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('C:\\Users\\Jatin\\Desktop\\deep-learning-face-detection\\deploy.prototxt.txt', 'C:\\Users\\Jatin\\Desktop\\deep-learning-face-detection\\res10_300x300_ssd_iter_140000.caffemodel')

# initializing the video stream
#print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(2.0)# to allow camera sensors to warm up
count=0
count1=0
# loop over the frames from the video stream
url='http://192.168.43.1:8080/shot.jpg'
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	imgres=urllib.request.urlopen(url)
	imgnp=np.array(bytearray(imgres.read()),dtype=np.uint8)
	frame=cv2.imdecode(imgnp,-1)
	#frame = vs.read()
	#time.sleep(0.5)#frames captured after an interval
	frame = cv2.resize(frame, (768,512))
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
 
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < 0.5:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		crop_img = frame[startY:endY, startX:endX]
		#cv2.imshow("cropped", crop_img) #can show the frame
		#crop_img=cv2.resize(crop_img,(193,260)) #can resize the image
		cv2.imwrite("C:\\Users\\Jatin\\Desktop\\Data Sets\\from video face\\frame%d.jpg" % count1, crop_img)
		count1=count1+1
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		

	# show the output frame
	cv2.imshow("Frame", frame)
	cv2.imwrite("C:\\Users\\Jatin\\Desktop\\Data Sets\\from video\\frame%d.jpg" % count, frame)
	count=count+1
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


cv2.destroyAllWindows()
#vs.stop()
