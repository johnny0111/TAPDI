import cv2 as cv
import Hough as ht
import imageForms as iF


def faceDetector(model, img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img,x,y,w,h

def main():
    scale = 1
    delta = 0
    ddepth = cv.CV_16U


    face_cascade = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    img = cv.imread('face.jpg')
    plaform, device, ctx, commQ, prog = ht.Setup()
    img,x,y,w,h = faceDetector(face_cascade,img)
    
    cv.imshow("fdw",img)


    #cv.imshow('img', img)
    cv.waitKey()


if __name__ == "__main__":
    main()
