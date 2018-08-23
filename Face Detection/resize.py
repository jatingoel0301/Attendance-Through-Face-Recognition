import os
import PIL
from PIL import Image

mywidth = 300

list=[]
img_path="C:\\Users\\Jatin\\Desktop\\Data Sets\\from video face"
for x in os.listdir(img_path):
    list.append(x)


for count in range(len(list)):
 img = Image.open('C:\\Users\\Jatin\\Desktop\\Data Sets\\from video face\\frame%d.jpg' % count)
 wpercent = (mywidth/float(img.size[0]))
 hsize = int((float(img.size[1])*float(wpercent)))
 img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
 img.save('C:\\Users\\Jatin\\Desktop\\Data Sets\\from video resize\\frame%d.jpg' % count)
