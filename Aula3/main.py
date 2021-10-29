import cv2 as cv
import numpy as np

import imageForms
import imageForms as iF
import ImageSegmentation as IS
import tkinter as tk

if __name__ == "__main__":
    root_1 = tk.Tk()
    root_1.withdraw()
    file_path_1 = tk.filedialog.askopenfilename()
    root_2 = tk.Tk()
    root_2.withdraw()
    file_path_2 = tk.filedialog.askopenfilename()
    img1 = cv.imread(file_path_1)
    img2 = cv.imread(file_path_2)



    # point 3
    imgLabels1 = IS.GetConnectedComponents(img1)
    imgLabels2 = IS.GetConnectedComponents(img2)

    imageForms.showSideBySideImages(img1,imgLabels1, "", False, False)
    imageForms.showSideBySideImages(img2,imgLabels2, "", False, False)

    img1[imgLabels1 == 0] = [255, 0, 0]
    img2[imgLabels2 == 0] = [255, 0, 0]

    imageForms.showSideBySideImages(img1,imgLabels1, "", False, False)
    imageForms.showSideBySideImages(img2,imgLabels2, "", False, False)


    #point 4
    img1_clean = cv.imread(file_path_1)
    img2_clean = cv.imread(file_path_2)
    KmeansImg1 = IS.Kmeans_Clustering(img1_clean, 3)
    KmeansImg2 = IS.Kmeans_Clustering(img2_clean, 3)

    imageForms.showSideBySideImages(img1_clean,KmeansImg1, "", False, False)
    imageForms.showSideBySideImages(img2_clean,KmeansImg2, "", False, False)


    #point 5
    filename = "Images/rice.bmp.mask.bmp"
    img_marks = cv.imread(filename)
    img11_clean = cv.imread(file_path_1)
    img12_clean = cv.imread(file_path_1)
    wt_marks = IS.GetWatershedFromMarks(img1_clean, img_marks)
    wt_noMarks = IS.GetWatershedByImmersion(img1_clean)
    img11_clean[wt_marks == 0] = [255, 0, 0]
    img12_clean[wt_noMarks == 0] = [255, 0, 0]
    imageForms.showSideBySideImages(img11_clean, img12_clean, "")

    #point 7
    img1_clean_7 = cv.imread(file_path_1)
    img2_clean_7 = cv.imread(file_path_1)
    img1_clean_7_blur = cv.blur(img1_clean_7, (5, 5))
    img2_clean_7_blur = cv.blur(img2_clean_7, (5, 5))
    wt_marks = IS.GetWatershedFromMarks(img1_clean, img_marks)
    wt_noMarks = IS.GetWatershedByImmersion(img1_clean)
    img11_clean[wt_marks == 0] = [255, 0, 0]
    img12_clean[wt_noMarks == 0] = [255, 0, 0]
    imageForms.showSideBySideImages(img11_clean, img12_clean, "")
