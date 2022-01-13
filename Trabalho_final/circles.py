import cv2 as cv
import Hough as ht
import aula9_sobel as sb
import imageForms as iF
import cv2
import numpy as np
def main():
    plaform, device, ctx, commQ, prog = ht.Setup()
    plaformS, deviceS, ctxS, commQS, progS = sb.sobelSetup()
    img = cv.imread('coinss.png')
    imgSobel = sb.sobelGPU(img,100,200,plaformS,deviceS,ctxS,commQS,progS)
    r_min = 10
    r_max = 100

    #cv2.imshow('Sobel X', img_gray)
   # img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    #imgSobel = cv.Sobel(img_blur, cv.CV_8U, 1, 0)

    # Sobel Edge Detection

    # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    #
    # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    #
    # imgSobel = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # ret, imgSobel = cv.threshold(imgSobel, 100, 255, cv.THRESH_BINARY)
    cv2.imshow('Sobel X', imgSobel)
    #
    circle = ht.Circle_Ht(imgSobel, r_min, r_max, plaform, device, ctx, commQ, prog)
    cv.circle(img, [circle[0],circle[1]], circle[2], (0, 0, 255), 3)
    cv2.imshow('Sobel X', img)
    # for i in accumulator:
    #     if(i != 0):
    #         print(i)
    # # Display Sobel Edge Detection Images
    #
   # cv2.imshow('Sobel X', imgSobel)
    #
    # cv2.waitKey(0)
    #
    # cv2.imshow('Sobel Y', sobely)
    #
    # cv2.waitKey(0)
    #
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    #
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

