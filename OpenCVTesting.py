from os import listdir
from os.path import isfile, join
import numpy
import cv2 as cv
# Loads all Files in A directory and Displays them one by one. Non image files will crash the system but that shouldn't be a problem

    

mypath='C:/Users/quinn/OneDrive/Pictures'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv.imread( join(mypath,onlyfiles[n]) )

for n in range(0, len(onlyfiles) - 1):
    cv.imshow('does the text here matter?', images[n])
    cv.waitKey(0)

cv.waitKey(0)