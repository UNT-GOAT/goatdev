from os import listdir
from os.path import isfile, join
import cv2
import os

# Open the default camera
cam0 = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam0.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam0.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

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
    elif command.lower() == 'q':
        break
    # Press 'q' to exit the loop
    

# Release the capture and writer objects
cam0.release()
out.release()
cv2.destroyAllWindows()