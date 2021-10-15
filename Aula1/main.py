# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from matplotlib import pyplot as plt

import huffman as h
import cv2 as cv
import math


def entropy_image(image, histogram):
    entropyB = 0

    for y in range(0, 255):
        print(histogram[y])
        if histogram[y] != 0:
            entropyB = entropyB + histogram[y] * math.log(histogram[y], 2)


def main():
    filename = "aula1.bmp"

    img = cv.imread(filename)

    cv.imshow("Imagem", img)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist), plt.xlim([0, 256])
    plt.show()
    huffman_dict = h.huffman(img.flatten())
    #print(huffman_dict)
    height, width, depth = img.shape
    enc = ""
    for x in range(0, height):
        for y in range(0, width):
            pixel = img[x, y]
            if pixel[0] == 7:
                #print("0111")
                enc += "0111"
            if pixel[0] == 69:
                #print("0110")
                enc += "0110"
            if pixel[0] == 147:
                #print("00")
                enc += "00"
            if pixel[0] == 221:
                #print("010")
                enc += "010"
            if pixel[0] == 255:
                #print("1")
                enc += "1"
    #print(len(enc))
    img_blur = cv.blur(img, [50, 50])
    print(entropy_image(img_blur,hist))
    cv.waitKey()


if __name__ == "__main__":
    main()
