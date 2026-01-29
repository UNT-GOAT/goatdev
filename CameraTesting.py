import cv2
import os
# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

numImages = 0
directory = 'Photos'


onlyfiles = [f for f in os.listdir(directory) if isfile(join(directory,f))]
for n in range(0, len(onlyfiles)):
    numImages += 1
numImages+= 1

os.chdir(directory)
while True:
    command = input("press 'c' to capture an image, 'q' to quit: ")
    if command.lower() == 'c':
        ret, frame = cam.read()
        tempString = 'img_' + str(numImages) + '.jpg'
        cv2.imwrite(tempString, frame)
        numImages += 1
    elif command.lower() == 'q':
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()