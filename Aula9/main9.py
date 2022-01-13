import cv2 as cv
import imageForms as iF
import aula9_sobel as sobel
import brightness as b
def main():
    img = cv.imread("faces.jpg")
    imgb = cv.imread("faces.jpg")
    plaform, device, ctx, commQ, prog = sobel.sobelSetup()
    img2 = sobel.sobelGPU(img, 50, 10,plaform,device,ctx,commQ,prog)
    iF.showSideBySideImages(img,img2,"IMG", False, False)

    plaform, device, ctx, commQ, prog = b.Setup()
    img3 = b.GPU(img, 0.5, 2,plaform,device,ctx,commQ,prog)
    iF.showSideBySideImages(imgb, img3, "IMG", False, False)
if __name__ == "__main__":
    main()