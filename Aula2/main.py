import JPEGCompression as jpeg
import cv2 as cv
import imageForms as iF

if __name__ == "__main__":
    filename = "usb_32x32.png"
    img = cv.imread(filename)
    cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    imgChannelY, imgChannelCb, imgChannelCr = cv.split(img)

    imgChannelC = cv.resize(img, None, fx=0.5, fy=1, interpolation=cv.INTER_AREA)

    for x in range(0, imgChannelY.shape[0], 8):
        for y in range(0, imgChannelY.shape[1], 8):
            imgChannelBlock = imgChannelY[x: x+8, y: y+8]
            result = jpeg.blockProcessing(imgChannelBlock, luminanceOrChrominance=1, compFactor=30)
            imgChannelY[x:x+8, y:y+8] = result

    for x in range(0, imgChannelCb.shape[0], 8):
        for y in range(0, imgChannelCb.shape[1], 8):
            imgChannelBlock = imgChannelCb[x: x+8, y: y+8]
            result = jpeg.blockProcessing(imgChannelBlock, luminanceOrChrominance=0, compFactor=30)
            imgChannelCb[x:x+8, y:y+8] = result

    for x in range(0, imgChannelCr.shape[0], 8):
        for y in range(0, imgChannelCr.shape[1], 8):
            imgChannelBlock = imgChannelCr[x: x+8, y: y+8]
            result = jpeg.blockProcessing(imgChannelBlock, luminanceOrChrominance=0, compFactor=30)
            imgChannelCr[x:x+8, y:y+8] = result

    imgChannelC = cv.resize(img, None, fx=2, fy=1, interpolation=cv.INTER_AREA)
    result = cv.merge((imgChannelY, imgChannelCb, imgChannelCr))
    result = cv.cvtColor(result, cv.COLOR_YCrCb2BGR)
    iF.showSideBySideImages(img, result, "", False, False)
