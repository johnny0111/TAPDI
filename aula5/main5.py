import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import ImageForms as IF
import tkinter as tk
import ImageFFT as FFT

#Ex1
root = tk.Tk()
root.withdraw()
file_path = tk.filedialog.askopenfilename()
img = cv.imread(file_path)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgMag, imgPhase = FFT.GetFFT_Mag_Phase(img_gray)
imgMagLog = cv2.log(imgMag)
imgPhaseLog = cv2.log(imgPhase)
imgMagLogShift = np.fft.fftshift(imgMagLog)
imgPhaseLogShift = np.fft.fftshift(imgPhaseLog)
IF.showSideBySideImages(imgMagLogShift, imgPhaseLogShift)
cv.waitKey()

#Ex2
filename1 = "aula5 (1).bmp"
filename2 = "aula5 (2).bmp"

img1 = cv.imread(filename1)
img2 = cv.imread(filename2)

img1Grey = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2Grey = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

img1MAG, img1PHASE = FFT.GetFFT_Mag_Phase(img1Grey)
img2MAG, img2PHASE = FFT.GetFFT_Mag_Phase(img2Grey)

img1Inverse = FFT.GetFFT_Inverse_Mag_Phase(img1MAG,img2PHASE)
img2Inverse = FFT.GetFFT_Inverse_Mag_Phase(img2MAG,img1PHASE)

IF.showSideBySideImages(img1Inverse, img2Inverse)


#Ex3
#img1MagLog = cv.log(img1MAG)
LpFilter = FFT.CreateFilterMask_Ideal(img1MAG.shape, 40, False)
result = cv2.multiply(img1MAG, np.fft.fftshift(LpFilter))
img1Inv = FFT.GetFFT_Inverse_Mag_Phase(result,img1PHASE)

HpFilter = FFT.CreateFilterMask_Ideal(img1MAG.shape, 40, True)
resultHp = cv2.multiply(img1MAG, np.fft.fftshift(HpFilter))
img1InvHp = FFT.GetFFT_Inverse_Mag_Phase(resultHp,img1PHASE)

IF.showSideBySideImages(img1InvHp, img1Inv)

#Ex4

LpGaussianFilter = FFT.CreateFilterMask_Gaussian(img1MAG.shape, 40, False)
HpGaussianFilter = FFT.CreateFilterMask_Gaussian(img1MAG.shape, 40, True)

resultLpG = cv2.multiply(img1MAG, np.fft.fftshift(LpGaussianFilter))
resultHpG = cv2.multiply(img1MAG, np.fft.fftshift(HpGaussianFilter))

img1InvLpG = FFT.GetFFT_Inverse_Mag_Phase(resultLpG,img1PHASE)
img1InvHpG = FFT.GetFFT_Inverse_Mag_Phase(resultHpG,img1PHASE)

IF.showSideBySideImages(img1InvLpG, img1InvHpG)