import cv2 as cv
import Hough as ht
import imageForms as iF


def faceDetector(model, img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

def main():
    scale = 1
    delta = 0
    ddepth = cv.CV_16U


    face_cascade = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    img = cv.imread('coins.png')
    plaform, device, ctx, commQ, prog = ht.Setup()




    #img = faceDetector(face_cascade,img)


    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.imshow('window_name', grad)


    #cv.imshow('img', img)
    cv.waitKey()


if __name__ == "__main__":
    main()
