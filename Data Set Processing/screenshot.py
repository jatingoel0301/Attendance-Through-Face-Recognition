import cv2
import time
vidcap = cv2.VideoCapture('C:\\Users\\Jatin\\Desktop\\Data Sets\\IMG_1830.MOV')
success,image = vidcap.read()
count = 0
success = True
while success:
  #time.sleep(0.5)
  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*500))
  cv2.imwrite("C:\\Users\\Jatin\\Desktop\\Data Sets\\Subject02\\Exhaustive\\Frame detected\\frame%d.jpg" % count, image)     
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count=count+1
