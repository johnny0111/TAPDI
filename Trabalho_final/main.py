import cv2 as cv
import imageForms as iF


def faceDetector(model, img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

def main():

    face_cascade = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    img = cv.imread('face.jpg')
    img = faceDetector(face_cascade,img)

    cv.imshow('img', img)
    cv.waitKey()


if __name__ == "__main__":
    main()
