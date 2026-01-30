import cv2
import os
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        camRun(self.camID)


def camRun(camID):             
    cam = cv2.VideoCapture(camID)


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    numImages = 0
    directory = 'Photos'

    print(os.getcwd())
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
    print(os.getcwd())
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
    elif command.lower == 'q':
        sys.exit
        print("trying to exit")