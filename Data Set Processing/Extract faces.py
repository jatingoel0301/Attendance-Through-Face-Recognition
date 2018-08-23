

# importing the necessary packages
import numpy as np
import cv2
import os


list=[]
img_path="C:\\Users\\Jatin\\Desktop\\Data Sets\\Subject02\\Exhaustive\\Frame detected"
for x in os.listdir(img_path):
    list.append(x)



for count in range(len(list)):
 # loading our serialized model
 #print("[INFO] loading model...")
 net = cv2.dnn.readNetFromCaffe('C:\\Users\\Jatin\\Desktop\\deep-learning-face-detection\\deploy.prototxt.txt', 'C:\\Users\\Jatin\\Desktop\\deep-learning-face-detection\\res10_300x300_ssd_iter_140000.caffemodel')

 # loading the input image and constructing an input blob for the image
 # by resizing to a fixed 300x300 pixel size and then normalizing it
 image = cv2.imread('C:\\Users\\Jatin\\Desktop\\Data Sets\\Subject02\\Exhaustive\\Frame detected\\frame%d.jpg' % count)
 (h, w) = image.shape[:2]
 blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
 	(300, 300), (104.0, 177.0, 123.0))

 # passing the blob through the network and obtain the detections and
 # predictions
 #print("[INFO] computing object detections...")
 net.setInput(blob)
 detections = net.forward()

 # looping over the detections
 for i in range(0, detections.shape[2]):
 	# extracting the confidence (i.e., probability) associated with the
	# prediction
	 confidence = detections[0, 0, i, 2]

	# filtering out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	 if confidence > 0.5:
		# computing the (x, y)-coordinates of the bounding box for the
		# object
		 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		 (startX, startY, endX, endY) = box.astype("int")

		 crop_img = image[startY:endY, startX:endX]
		 #cv2.imshow("cropped", crop_img)
		 cv2.imwrite("C:\\Users\\Jatin\\Desktop\\Data Sets\\Subject02\\Exhaustive\\Face cropped\\frame%d.jpg" % count, crop_img)
 
		# drawing the bounding box of the face along with the associated
		# probability
		 text = "{:.2f}%".format(confidence * 100)
		 y = startY - 10 if startY - 10 > 10 else startY + 10
		 cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		 cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


 # showing the output image
 #print(image.shape[0]," ",image.shape[1])
 image=cv2.resize(image,(768,512))
 #print(image.shape[0]," ",image.shape[1])
 #cv2.imshow("Output", image)
 #cv2.imwrite("C:\\Users\\Jatin\\Desktop\\Data Sets\\Subject02\\Exhaustive\\Frame detected\\frame%d.jpg" % count, image)
print("The faces have been cropped")
cv2.waitKey(0)
