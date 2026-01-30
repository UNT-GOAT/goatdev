import cv2
import os
import threading
import time

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        camRun(self.camID)

class camCalibration(threading.Thread):
    def __init__(self, camID):
        threading.Thread.__init__(self)
        self.camID = camID
    def run(self):
        camCalibrate(self.camID)


def camCalibrate(camID):
    cam = cv2.VideoCapture(camID)
    
    while True: 
        ret, frame = cam.read()
        cv2.imshow("window", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows
    

def camRun(camID):             
    cam = cv2.VideoCapture(camID)


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    numImages = 0
    directory = 'Photos'

    onlyfiles = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    for n in range(0, len(onlyfiles)):
        numImages += 1
    numImages+= 1

    os.chdir(directory)
    
    ret, frame = cam.read()
    tempString = 'camera' + str(camID) + 'img_' + str(numImages) + '.jpg'
    cv2.imwrite(tempString, frame)
    numImages += 1
    
    #change directory back to root
    os.chdir("..") 
    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()
    
    


while True:
    command = input("press 'c' to capture an image, 'q' to quit: ")
    if command.lower() == 'c':
        threads = [
        camThread("Camera 1", 0),
        camThread("Camera 2", 1),
        ]
        for t in threads:
            t.start()
        #wait for threads to finish
        for t in threads:
            t.join()
    elif command.lower() == 'q':
        break