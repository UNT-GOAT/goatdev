from os import listdir
from os.path import isfile, join
import cv2
import os

# Open the default camera
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(2)

numImages = 0 #images captured thus far

#directory for captured images
directory = 'CameraTesting'
os.chdir(directory)

onlyfiles = [ f for f in listdir(directory) if isfile(join(directory,f)) ]
for n in range(0, len(onlyfiles)):
    numImages += 1
numImages += 1

while True:
    command = input("Press 'c' to capture an image, 'q' to quit: ")
    if command.lower() == 'c':
        ret, frame = cam0.read()
        tempString = 'img_' + str(numImages) + '.jpg'
        cv2.imwrite(tempString, frame)
        numImages += 1

        ret, frame = cam1.read()
        tempString = 'img_' + str(numImages) + '.jpg'
        cv2.imwrite(tempString, frame)
        numImages += 1

        ret, frame = cam2.read()
        tempString = 'img_' + str(numImages) + '.jpg'
        cv2.imwrite(tempString, frame)
        numImages += 1
    elif command.lower() == 'q':
        break
    # Press 'q' to exit the loop
    

# Release the capture and writer objects
cam0.release()
cam1.release()
cam2.release()

cv2.destroyAllWindows()