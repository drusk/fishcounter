import cv
import numpy

IMAGE="data/fish_ss.png"

img = cv.LoadImageM(IMAGE, cv.CV_LOAD_IMAGE_GRAYSCALE)
eig_image = cv.CreateMat(img.rows, img.cols, cv.CV_32FC1)
temp_image = cv.CreateMat(img.rows, img.cols, cv.CV_32FC1)
for (x,y) in cv.GoodFeaturesToTrack(img, eig_image, temp_image, 15, 0.54, 1.0, useHarris = False):
    print "good feature at", x,y
    cv.Circle(img, (int(x),int(y)), 7, cv.RGB(250, 7, 10), 2)

cv.ShowImage("foo", img)
cv.WaitKey()
