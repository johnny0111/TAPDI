import cv2

import ImageHough as iH
from matplotlib import pyplot as plt
import cv2 as cv
import math
import ImageForms as iF
import tkinter as tk

filename = "aula4-3.bmp"

#Ex1
img = cv.imread(filename)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
accumulator, thetas, rhos = iH.HoughPlane(img_gray, 0, 180, 1)
iF.showSideBySideImages(img, accumulator, "")

#Ex2
img_lines = iH.ShowHoughLines(img_gray, img, 3)
iF.showSideBySideImages(img, img_lines, "")

#Ex3
img2 = cv.imread("aula4-2.bmp")
img_gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
imgSobel = cv.Sobel(img_gray2,cv.CV_8U,1,0)
ret,imgW = cv.threshold(imgSobel, 255, 255, cv.THRESH_OTSU + cv.THRESH_BINARY, imgSobel)
img_lines_sobel = iH.ShowHoughLines(imgW, img2, 120)
iF.showSideBySideImages(img_lines_sobel, imgW, "")

#Ex4
img_lines_segment = iH.ShowHoughLineSegments(imgW, img2, 120)
iF.showSideBySideImages(img_lines_segment, imgW, "")
 
#Ex5
img3 = cv.imread("aula4-coins.png")
img_gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
img_lines_circles = iH.ShowHoughCircles(img_gray3, img3, 120)
iF.showSideBySideImages(img3, img_lines_circles, "", False, False)

#Ex6
root = tk.Tk()
root.withdraw()
file_path = tk.filedialog.askopenfilename()
cap = cv2.VideoCapture(file_path)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        img_gray6 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        imgSobel6 = cv.Sobel(img_gray6, cv.CV_8U, 1, 0)
        ret6, imgW6 = cv.threshold(imgSobel6, 255, 255, cv.THRESH_OTSU + cv.THRESH_BINARY, imgSobel6)
        img_lines_segment6 = iH.ShowHoughLineSegments(imgW6, frame, 120)
        cv2.imshow('Frame',img_lines_segment6)
    if (cv.waitKey(20) >= 0):
        break
cap.release()