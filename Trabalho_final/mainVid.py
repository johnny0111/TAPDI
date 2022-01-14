import cv2
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
    f = open("Parameters_Set.txt", "r")
    lines = []
    for line in f:
        if 'Threshold(t1):' in line:
            t1 = line.strip("Threshold(t1): ")
        elif 'Threshold(t2):' in line:
            t2 = line.strip("Threshold(t2): ")
        elif 'RadiusMax:' in line:
            r_max = line.strip("RadiusMax: ")
        elif 'RadiusMin:' in line:
            r_min = line.strip("RadiusMin: ")
    f.close()
    scale = 1
    delta = 0
    ddepth = cv.CV_16U
    # r_min = 10
    # r_max = 100
    # t1 = 10
    # t2 = 100

    vid = cv2.VideoCapture(0)
    while(1):
        x = 0
        y = 0
        w = 0
        h = 0
        ret, img = vid.read()
        #cv2.imshow()


        face_cascade = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        #img = cv.imread('face.jpg')
        img_original = img.copy()
        plaformHt, deviceHt, ctxHt, commQHt, progHt = hT.Setup()
        plaformS, deviceS, ctxS, commQS, progS =  sB.sobelSetup()

        img,x,y,w,h = faceDetector(face_cascade,img)
        #img_sobel, x, y, w, h = faceDetector(face_cascade, img)
        img_left = img_original[y:round(h/2)+y,x:round(w/2) +x]
        img_rigt = img_original[y:round(h/2)+y, x+round(w/2):x+w]
        cv.cvtColor(img_left, cv.COLOR_BGR2BGRA)
        cv.cvtColor(img_rigt, cv.COLOR_BGR2BGRA)
        cv.cvtColor(img,cv.COLOR_BGR2BGRA)

        img_sobel = sB.sobelGPU(img_original,t1,t2,plaformS,deviceS,ctxS,commQS,progS)
        img_left_sobel = sB.sobelGPU(img_left,t1,t2,plaformS,deviceS,ctxS,commQS,progS)
        img_right_sobel = sB.sobelGPU(img_rigt, t1, t2, plaformS, deviceS, ctxS, commQS, progS)
        circle_left = hT.Circle_Ht(img_left_sobel, r_min, r_max, plaformHt, deviceHt, ctxHt, commQHt, progHt)
        circle_right = hT.Circle_Ht(img_right_sobel, r_min, r_max, plaformHt, deviceHt, ctxHt, commQHt, progHt)

        cv.circle(img_sobel, [circle_left[0] + x, circle_left[1] +y], circle_left[2], (0, 255, 0), 3)
        cv.circle(img_sobel, [circle_right[0] + x + round(w/2) , circle_right[1] + y], circle_right[2], (0, 0, 255), 3)
        cv.circle(img, [circle_left[0] + x, circle_left[1] +y], circle_left[2], (0, 255, 0), 3)
        cv.circle(img, [circle_right[0] + x + round(w/2) , circle_right[1] + y], circle_right[2], (0, 0, 255), 3)
        #iF.showSideBySideImages(img_sobel, img, "img",False,False)


        #cv.imshow('img', img_sobel)
        cv.imshow('img', img)
        #cv.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()


if __name__ == "__main__":
    main()