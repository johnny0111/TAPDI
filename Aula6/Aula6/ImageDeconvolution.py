
import cv2 as cv
import numpy as np
import imageForms as iF
import math

def GetFFT_Re_Im(imgGray):
    """
    Get FFT Real and Imaginary components
    :param imgGray:
    :return: Real and Imaginary components
    """

    rows, cols = imgGray.shape
    m = cv.getOptimalDFTSize( rows )
    n = cv.getOptimalDFTSize( cols )
    imgGray_padded = cv.copyMakeBorder(imgGray, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
    imgGray_padded = np.float32(imgGray_padded)
    planes = [np.float32(imgGray_padded), np.zeros(imgGray_padded.shape, np.float32)]
    imgComplex_planes = cv.merge(planes)  # Add to the expanded another plane with zeros

    imgComplex_planes = cv.dft(imgGray_padded,flags=cv.DFT_COMPLEX_OUTPUT )

    cv.split(imgComplex_planes, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

    #return imgComplex_planes
    return planes[0], planes[1]

def GetFFT_Inverse_Re_Im(imgRe, imgIm):
    """
    Get inverse FFT Real and Imaginary components
    :param imgRe:
    :param imgIm:
    :return: image gray
    """
    planes = [imgRe, imgIm]
    complexI = cv.merge(planes)  # Add to the expanded another plane with zeros

    complexI= cv.idft(complexI, flags=cv.DFT_SCALE)

    cv.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    imgRes = planes[0]
    cv.normalize(imgRes,imgRes,0,255,cv.NORM_MINMAX)
    return imgRes

def CreateFilterMask_Gaussian(shape, cutOff, HighPass, order=1):
    """
    Create a Gaussian Butterworth filter mask
    :param shape:
    :param cutOff:
    :param HighPass:
    :param order:
    :return: mask
    """
    width, height = shape
    mask = np.zeros(shape, np.float32)

    for u in range(0, width):
        for v in range(0, height):
            uu = u - width / 2.0
            vv = v - height / 2.0
            if (HighPass):
                mask[u, v] = 1 - 1.0 / (1 + ((math.sqrt(2) - 1) * math.pow(math.sqrt(uu * uu + vv * vv) / (cutOff), 2*order)))

            else:
                mask[u, v] = 1.0 / (1+ ((math.sqrt(2)-1) * math.pow( math.sqrt(uu * uu + vv * vv) / (cutOff), 2 *order )))

    return mask

def GetFilterConv(motion, cutOff):
    """
    Create a gaussian / motion filter
    :param motion:
    :param cutOff:
    :return:
    """
    if motion == False:
        filter = CreateFilterMask_Gaussian((25,25),cutOff,False)
        filter = filter/np.sum(filter)


    else :
        filter =np.array( [[0,0,0,0,0,0,0,0,0,0,0,0,0,0.013074,0.026806],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0.013074,0.044639,0.013074],
                            [0,0,0,0,0,0,0,0,0,0,0,0.013074,0.044639,0.013074,0],
                            [0,0,0,0,0,0,0,0,0,0,0.013074,0.044639,0.013074,0,0],
                            [0,0,0,0,0,0,0,0,0,0.013074,0.044639,0.013074,0,0,0],
                            [0,0,0,0,0,0,0,0,0.013074,0.044639,0.013074,0,0,0,0],
                            [0,0,0,0,0,0,0,0.013074,0.044639,0.013074,0,0,0,0,0],
                            [0,0,0,0,0,0,0.013074,0.044639,0.013074,0,0,0,0,0,0],
                            [0,0,0,0,0,0.013074,0.044639,0.013074,0,0,0,0,0,0,0],
                            [0,0,0,0,0.013074,0.044639,0.013074,0,0,0,0,0,0,0,0],
                            [0,0,0,0.013074,0.044639,0.013074,0,0,0,0,0,0,0,0,0],
                            [0,0,0.013074,0.044639,0.013074,0,0,0,0,0,0,0,0,0,0],
                            [0,0.013074,0.044639,0.013074,0,0,0,0,0,0,0,0,0,0,0],
                            [0.013074,0.044639,0.013074,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0.026806,0.013074,0,0,0,0,0,0,0,0,0,0,0,0,0]])

    return filter

def DivideComplex(numRe_a,numIm_b, denRe_c, denIm_d):
    """
    Divide two Complex matrices
    :param imgFFT_Re:
    :param imgFFT_Im:
    :param filterPadded:
    :return: complex
    """
    denRes = np.power(denRe_c,2)+np.power(denIm_d,2)

    num1 = np.multiply (numRe_a, denRe_c) + np.multiply(numIm_b, denIm_d)
    num2 = np.multiply(numIm_b, denRe_c) - np.multiply(numRe_a, denIm_d)

    imgResRe = np.divide(num1, denRes, out=np.zeros_like(num1), where=denRes!=0)
    imgResIm = np.divide(num2, denRes, out=np.zeros_like(num2), where=denRes!=0)

    return imgResRe, imgResIm

def InverseDeconvolution(imgRes, motion, cutOff):
    """
    Simple Inverse Deconvolution G/H
    :param imgRes: grayscale image
    :param motion: True or False
    :param cutOff: filter cutOff frequency
    :return: restored image
    """
    imgFFT_Re, imgFFT_Im = GetFFT_Re_Im(imgRes)

    # prepare filter - padded with zeros
    filter = GetFilterConv(motion, cutOff)

    widthDiff = (imgFFT_Re.shape[0] - filter.shape[0])
    heightDiff = (imgFFT_Re.shape[1] - filter.shape[1])
    left = np.int32(np.round(widthDiff/2))
    right = np.int32(widthDiff - left)
    top  = np.int32(np.round( heightDiff/2))
    bottom = np.int32(heightDiff-top)

    filterPadded = cv.copyMakeBorder(filter, top, bottom,left,right, cv.BORDER_CONSTANT, value=[0, 0, 0])
    filterFFT_Re, filterFFT_Im = GetFFT_Re_Im(filterPadded)

    imgResRe, imgResIm =   DivideComplex(imgFFT_Re, imgFFT_Im, filterFFT_Re, filterFFT_Im)

    imgResInv = GetFFT_Inverse_Re_Im(imgResRe, imgResIm)

    return np.fft.fftshift(imgResInv)

def InverseDeconvolutionButterworth(imgRes, motion, cutOff,cutOffBW = 50):
    """
    Inverse Deconvolution with Butterworth filtering
    :param imgRes: grayscale image
    :param motion: True or False
    :param cutOff: filter cutOff frequency
    :param cutOffBW: Butterworth filter cutOff frequency

    :return: restored image
    """

    #get image FFT
    imgFFT_Re, imgFFT_Im = GetFFT_Re_Im(imgRes)
    planes = [imgFFT_Re, imgFFT_Im]
    imgFFT_Complex = cv.merge(planes)

    # prepare filter - padded with zeros
    filter = GetFilterConv(motion, cutOff) #get filter

    widthDiff = (imgFFT_Re.shape[0] - filter.shape[0])
    heightDiff = (imgFFT_Re.shape[1] - filter.shape[1])
    left = np.int32(np.round(widthDiff/2))
    right = np.int32(widthDiff - left)
    top  = np.int32(np.round( heightDiff/2))
    bottom = np.int32(heightDiff-top)
    filterPadded = cv.copyMakeBorder(filter, top, bottom,left,right, cv.BORDER_CONSTANT, value=[0, 0, 0])

    filterFFT_Re, filterFFT_Im = GetFFT_Re_Im(filterPadded) #get filter FFT

    #Buttwerworth
    filterBW = CreateFilterMask_Gaussian(filterFFT_Re.shape,cutOffBW,False,4)
    filterBW_Re = filterBW/np.sum(filterBW)
    filterBW_Re = np.fft.fftshift(filterBW_Re)
    filterBW_Im = np.zeros(filterBW_Re.shape)

    # T(u, v) = B(u, v) / H(u, v)
    imgFilterRe, imgFilterIm =   DivideComplex(filterBW_Re, filterBW_Im, filterFFT_Re, filterFFT_Im)
    planes = [imgFilterRe, imgFilterIm]
    planes = np.float32(planes)
    imgFilter_Complex =cv.merge(planes)

    # G(u,v) * T(u,v)
    imgRes = cv.mulSpectrums(imgFFT_Complex, imgFilter_Complex, flags=cv.DFT_ROWS)
    cv.split(imgRes, planes)

    # Get restored image
    imgResInv = GetFFT_Inverse_Re_Im(planes[0],planes[1])

    return np.fft.fftshift(imgResInv)

def InverseDeconvolutionWiener(imgInput, kFactor, motion, cutOff):
    """
    Inverse Deconvolution with Wiener filtering
    :param imgRes: grayscale image
    :param kFactor: WIener SNR
    :param motion: True or False
    :param cutOff: filter cutOff frequency

    :return: restored image
    """
    # get image FFT
    imgFFT_Re, imgFFT_Im = GetFFT_Re_Im(imgInput)
    planes = [imgFFT_Re, imgFFT_Im]
    imgFFT_G_Complex = cv.merge(planes)

    # prepare filter - padded with zeros
    filter = GetFilterConv(motion, cutOff)

    widthDiff = (imgFFT_Re.shape[0] - filter.shape[0])
    heightDiff = (imgFFT_Re.shape[1] - filter.shape[1])
    left = np.int32(np.round(widthDiff / 2))
    right = np.int32(widthDiff - left)
    top = np.int32(np.round(heightDiff / 2))
    bottom = np.int32(heightDiff - top)
    filterPadded = cv.copyMakeBorder(filter, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])

    filterFFT_H_Re, filterFFT_H_Im = GetFFT_Re_Im(filterPadded)
    planesFilter = [(filterFFT_H_Re),(filterFFT_H_Im)]
    filterFFT_H_Complex = cv.merge(planesFilter)

    #  G(u, v) * H(u, v)^
    num = cv.mulSpectrums(imgFFT_G_Complex, filterFFT_H_Complex, flags=cv.DFT_ROWS,conjB=True)

    # | H(u, v) | ^ 2
    den = cv.mulSpectrums( filterFFT_H_Complex, filterFFT_H_Complex, flags=cv.DFT_ROWS,conjB=True)

    # | H(u, v) | ^ 2 + SNR
    den = den + kFactor

    # H(u, v) *.G(u, v) / (| H(u, v) | ^ 2 + K)
    denPlanes = np.copy(planes)
    numPlanes = np.copy(planesFilter)
    cv.split(den, denPlanes)
    cv.split(num, numPlanes)

    imgResRe, imgResIm = DivideComplex(numPlanes[0],numPlanes[1], denPlanes[0], denPlanes[1])

    #Get restored image
    imgRes=GetFFT_Inverse_Re_Im(imgResRe, imgResIm )

    imgRes = np.fft.fftshift(imgRes)
    cv.normalize(imgRes, imgRes, 0, 255, norm_type=cv.NORM_MINMAX)
    return imgRes

# ############################################################################
# ############################################################################
# ############################################################################

#Open image
#pathname = "..\\Aula 6 - Metodos de Refocagem\\images\\"

#exercicio 1
#filename = "aula 6 (1).png"
#imgOriginal = cv.imread(pathname + filename)
#if (imgOriginal is None):
#    print("Image File Not Found")
 #   exit(-1)
#imgGray = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY)

#configs aula 6 (1)
cutOff = 2
cutOffBW = 50
noiseAmp = 10 #10%
kFactor = 0.001
motionSetting = False

"""
#configs aula 6 (2)
cutOff = 1
cutOffBW = 10
noiseAmp = 10 #10%
kFactor = 0.001
motionSetting = True
"""

# blur
#motion=False

#iF.showSideBySideImages(imgOriginal,imgResGauss,"Gauss Blur")

# motion blur
#motion=True

#if (motionSetting):
#    imgResGauss=imgResMotion
#iF.showSideBySideImages(imgOriginal,imgResMotion, "Motion Blur")

# add noise

#iF.showSideBySideImages(imgOriginal,imgResNoise, "Add noise")


#deconv with error


#deconv with Butterworth with error


#deconv with Wiener


