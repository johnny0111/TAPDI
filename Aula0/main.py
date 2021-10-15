from typing import Union, Any

import cv2 as cv
import tkinter as tk
from tkinter import filedialog
import imageForms as IF

# abrir imagem

pathname = "..\\Aula0\\images\\"
filename = "peppers.jpg"

img = cv.imread(pathname + filename)
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, imgBW = cv.threshold(imgGray, 12, 255, cv.THRESH_BINARY, cv.THRESH_OTSU)
img_blur = cv.blur(img, [10, 10])
cv.imshow("Imagem", imgBW)

IF.showSideBySideImages(img, img_blur, "", False, False)

brightness = float(input("Enter brightness\n"))
contrast = float(input("Enter contrast\n"))
height, width, depth = img.shape
for x in range(0, height):
    for y in range(0, width):
        pixel = img[x, y]
        new_pixel = pixel * contrast + brightness
        if new_pixel[0] > 255:
            new_pixel[0] = 255
        if new_pixel[1] > 255:
            new_pixel[1] = 255
        if new_pixel[2] > 255:
            new_pixel[2] = 255
        if new_pixel[0] < 0:
            new_pixel[0] = 0
        if new_pixel[1] < 0:
            new_pixel[1] = 0
        if new_pixel[2] < 0:
            new_pixel[2] = 0
        img[x, y] = new_pixel

cv.imshow("Imagem", img)

cv.waitKey()

"""root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

"""
# ret,imgBW = cv.threshold(imgGray,12,255, cv.THRESH_BINARY )
