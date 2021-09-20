#face-dectection #Haar

import time
import os
from matplotlib import pyplot
import cv2

faces = []
faceDetect = []
count = 0

start = time.time()
print("Record timer")


# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

folder_name = "path"
picture_set = os.listdir(folder_name)
for i in picture_set:
  picture = folder_name + '/' + str(i)
  pixels = pyplot.imread(picture)
  pyplot.imshow(pixels)
  pyplot.show()

  # Convert color image to grayscale 
  gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
  # detect faces in the image
  results = face_cascade.detectMultiScale(gray, 1.3, 5)

  count += 1

  for (x1, y1, width, height) in results:
    if y1<0:
      y1=0
    if x1<0:
      x1=0
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    pyplot.imshow(face)
    #im_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    #faceDetect.append({i+str(count):im_rgb})
    #cv2.imwrite('/content/drive/MyDrive/CP491/Lab4/Haar/Correct/'+i+'.png',im_rgb)
    pyplot.show()
  faces.append(face)

elapsed = (time.time() - start)
print('Performance ', elapsed)