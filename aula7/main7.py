import cv2
import cv2 as cv
import imageForms as iF
import numpy as np
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

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor.getDefaultPeopleDetector())


pedestrians = hog.detectMultiScale(img_pedestrians)

# Cycle through list of pedestrians found and draw rectangles
for (x,y,w,h) in pedestrians[0]:
    center = (x + w//2, y + h//2)
    img_pedestrians = cv.rectangle(img_pedestrians, (x,y), (x+w,y+h), (0, 0, 255), 4)

iF.showSideBySideImages(img_pedestrians_original, img_pedestrians, "HOG: original vs. pedestrians found")





