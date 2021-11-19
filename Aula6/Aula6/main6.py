import cv2
import cv2 as cv
import imageForms as iF
import ImageDeconvolution as iD
import numpy as np

#Ex2
file_path="Aula 6 (1).png"
img = cv.imread(file_path)

#configs aula 6 (1)
cutOff = 0.5
cutOffBW = 40
noiseAmp = 10 #10%
kFactor = 0.001
motionSetting = False

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
filter = iD.GetFilterConv(motionSetting, cutOff)
imgResGauss = cv.filter2D(img_gray,cv.CV_8U, filter, anchor=(np.int32(filter.shape[0]/2), np.int32(filter.shape[1]/2)))
#iF.showSideBySideImages(img, imgResGauss, "Filtro Gaussiano", False, False)

#Ex.3
imgRand = np.random.rand(img.shape[0], img.shape[1]) * noiseAmp
imgResGauss_Noise = imgResGauss + imgRand
#iF.showSideBySideImages(img, imgResGauss_Noise, "Imagem com ruido")

#Ex.4 (1ºmétodo)
img_Recovered = iD.InverseDeconvolution(imgResGauss, motionSetting, cutOff)
#iF.showSideBySideImages(imgResGauss, img_Recovered, "Imagem Desconvolucionada")
img_Recovered_Noise = iD.InverseDeconvolution(imgResGauss_Noise, motionSetting, cutOff)
#iF.showSideBySideImages(imgResGauss_Noise, img_Recovered_Noise, "Imagem com ruido Desconvoluncionada")
#Ex.4 (2ºmétodo)
img_Recovered_Butter = iD.InverseDeconvolutionButterworth(imgResGauss_Noise, motionSetting, cutOff, cutOffBW)
#iF.showSideBySideImages(imgResGauss, cv2.convertScaleAbs( img_Recovered_Butter, alpha=4, beta=-400), "Imagem Desconvolucionada com Butterworth")

#Ex.5
#configs aula 6 (2)
cutOff = 1
cutOffBW = 10
noiseAmp = 10 #10%
kFactor = 0.001
motionSetting = False

filename = "Aula 6 (2).png"
img2 = cv.imread(filename)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

filter2 = iD.GetFilterConv(motionSetting, cutOff)
imgResGauss2 = cv.filter2D(img2_gray,cv.CV_8U, filter2, anchor=(np.int32(filter2.shape[0]/2), np.int32(filter2.shape[1]/2)))
imgRand2 = np.random.rand(img2.shape[0], img2.shape[1]) * noiseAmp
imgResGauss2_Noise = imgResGauss2 + imgRand2

img_Wiener = iD.InverseDeconvolutionWiener(imgResGauss2, kFactor, motionSetting, cutOff)
iF.showSideBySideImages(imgResGauss2, img_Wiener, "Deconvolucao por Wiener")

img_Wiener_Noise = iD.InverseDeconvolutionWiener(imgResGauss2_Noise, kFactor, motionSetting, cutOff)
iF.showSideBySideImages(imgResGauss2_Noise, img_Wiener_Noise, "Deconvolucao por Wiener c/ruido")

motionSetting=True

filter3 = iD.GetFilterConv(motionSetting, cutOff)
imgResBlur = cv.filter2D(img2_gray,cv.CV_8U, filter3, anchor=(np.int32(filter3.shape[0]/2), np.int32(filter3.shape[1]/2)))
imgResBlur_Noise = imgResBlur + imgRand2

img_Wiener2 = iD.InverseDeconvolutionWiener(imgResBlur, kFactor, motionSetting, cutOff)
iF.showSideBySideImages(imgResBlur, img_Wiener2, "Deconvolucao por Wiener")

img_Wiener_Noise2 = iD.InverseDeconvolutionWiener(imgResBlur_Noise, kFactor, motionSetting, cutOff)
iF.showSideBySideImages(imgResBlur_Noise, img_Wiener_Noise2, "Deconvolucao por Wiener c/ruido")
