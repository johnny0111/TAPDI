import cv2
import cv2 as cv
import imageForms as iF
import numpy as np


def getFaceBox(frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Open DNN model
    modelFile = "models\\opencv_face_detector_uint8.pb"
    configFile = "models\\opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    # prepare for DNN
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (500, 300), [104,
                                                                   117, 123], True, False)

    # set image as DNN input
    net.setInput(blob)

    # get Output
    detections = net.forward()

    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0),
                          int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes

def GetHOGPedestrians_Detection(frame):
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor.getDefaultPeopleDetector())


    pedestrians = hog.detectMultiScale(frame)

    # Cycle through list of pedestrians found and draw rectangles
    for (x,y,w,h) in pedestrians[0]:
        center = (x + w//2, y + h//2)
        pedestrian = cv.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 4)
        return pedestrian

def main():
    """"
    faces = []
    template = []
    roads = []
    pedestrian = []

    file_path = "car template (1).bmp"
    template.append(cv.imread(file_path))
    file_path = "car template (2).bmp"
    template.append(cv.imread(file_path))

    file_path = "road (1).bmp"
    roads.append(cv.imread(file_path))
    file_path = "road (2).bmp"
    roads.append(cv.imread(file_path))
    file_path = "road (3).bmp"
    roads.append(cv.imread(file_path))
    file_path = "road (4).bmp"
    roads.append(cv.imread(file_path))
    file_path = "road (5).bmp"
    roads.append(cv.imread(file_path))

    file_path = "faces (1).jpg"
    faces.append(cv.imread(file_path))
    file_path = "faces (2).jpg"
    faces.append(cv.imread(file_path))
    file_path = "faces (3).png"
    faces.append(cv.imread(file_path))
    file_path = "faces (4).jpg"
    faces.append(cv.imread(file_path))
    file_path = "faces (5).jpg"
    faces.append(cv.imread(file_path))
    file_path = "faces (6).jpg"
    faces.append(cv.imread(file_path))
    file_path = "faces (7).jpg"
    faces.append(cv.imread(file_path))

    file_path = "pedestrian (1).jpg"
    pedestrian.append(cv.imread(file_path))
    file_path = "pedestrian (2).jpg"
    pedestrian.append(cv.imread(file_path))

    methods = ['cv.TM_SQDIFF_NORMED', 'cv.TM_CCORR_NORMED', 'cv.TM_CCOEFF_NORMED']

    #SumSquareDifference
    for t in template:
       template_gray = cv.cvtColor(t, cv.COLOR_BGR2GRAY)
       for r in roads:
           img_gray = cv.cvtColor(r, cv.COLOR_BGR2GRAY)
           dontprint=0
           for meth in methods:
               method = eval(meth)
               res = cv.matchTemplate(img_gray, template_gray, method)
               min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
               print(meth, min_val, max_val)
               img_grayCP = img_gray.copy()
               if meth == 'cv.TM_SQDIFF_NORMED':
                   cv.rectangle(img_grayCP, min_loc, (min_loc[0]+template_gray.shape[1], min_loc[1]+template_gray.shape[0]), color=(0, 255, 0), thickness=2)
               else:
                   cv.rectangle(img_grayCP, max_loc, (max_loc[0]+template_gray.shape[1], max_loc[1]+template_gray.shape[0]), color=(0, 255, 0), thickness=2)
               if meth == 'cv.TM_CCOEFF_NORMED':
                   if max_val < 0.33:
                       dontprint = 1
           if dontprint ==0:
               iF.showSideBySideImages(img_gray, img_grayCP, "Ex.1")


"""
    #ex2

    filename3 = "faces (1).jpg"
    img_faces = cv.imread(filename3)

    img_facesGray = cv.cvtColor(img_faces, cv.COLOR_BGR2GRAY)
    w= img_facesGray.shape[0]
    h = img_facesGray.shape[1]
    # Get pre-trained classifier
    haar = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # Apply Viola-Jones
    faces = haar.detectMultiScale(img_facesGray, scaleFactor = 1.4, minSize = (24, 24), maxSize = (img_facesGray.shape[1] // 2, img_facesGray.shape[0] // 2))
    # Cycle through faces list
    for(x, y, w, h) in faces:
        img_faces = cv.rectangle(img_faces, (x,y), (x+w,y+h), (0,0,255), 2)

    iF.showSideBySideImages(img_faces, img_faces, "Viola-Jones: resultado vs. resultado com BGR1=False", BGR1=False)

    #ex3

    filename4 = "pedestrian (1).jpg"
    img_pedestrians = cv.imread( filename4)
    img_pedestrians_original = cv.imread( filename4)


    img_pedestrians = GetHOGPedestrians_Detection(img_pedestrians)
    iF.showSideBySideImages(img_pedestrians_original, img_pedestrians, "HOG: original vs. pedestrians found")

    #ex4

    filename5 = "faces (7).jpg"
    img_faces2 = cv.imread(filename5)
    frame_face , box = getFaceBox(img_faces2)

    iF.showSideBySideImages(frame_face, img_faces2, "DNN", BGR1=False)

     #ex5

    filename6 = "pedestrian Video.mp4"
    vidCap = cv.VideoCapture(filename6)

    if (not vidCap.isOpened()):
        print("Video File Not Found")
        exit(-1)

    while (True):
        ret, vidFrame = vidCap.read()
        if (not ret):
            break

        imgOut = GetHOGPedestrians_Detection(vidFrame)

        cv.imshow("Video", imgOut)
        if (cv.waitKey(20) >= 0):
            break

if __name__ == "__main__":
    main()


