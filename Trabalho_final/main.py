import cv2 as cv
import Hough as hT
import imageForms as iF
import aula9_sobel as sB

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
    img_original = img.copy()
    plaformHt, deviceHt, ctxHt, commQHt, progHt = hT.Setup()
    plaformS, deviceS, ctxS, commQS, progS =  sB.sobelSetup()

    img,x,y,w,h = faceDetector(face_cascade,img)
    img_left = img_original[y:round(h/2)+y,x:round(w/2) +x]
    cv.cvtColor(img_left, cv.COLOR_BGR2BGRA)
    img_left_sobel = sB.sobelGPU(img_left,20,50,plaformS,deviceS,ctxS,commQS,progS)

    iF.showSideBySideImages(img_left_sobel, img_left, "img",False,False)


    #cv.imshow('img', img)
    cv.waitKey()


if __name__ == "__main__":
    main()
