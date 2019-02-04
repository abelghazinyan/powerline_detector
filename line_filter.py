import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

def nothing(x):
    pass

# kernel3 = np.real(gabor_kernel(1/18, theta=np.deg2rad(20), sigma_x=3.5, sigma_y=3.5))

# plt.imshow(kernel3, cmap='gray')
# plt.show()

# thresh = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

img = cv.imread("2.JPG")

cv.imshow('img', img)

accum = np.zeros_like(img)

cv.namedWindow('image')

# #create trackbars for params
cv.createTrackbar('ksize','image',10,100,nothing)
cv.createTrackbar('Sigma','image',50,100,nothing)
cv.createTrackbar('Theta','image',28,100,nothing)
cv.createTrackbar('f','image',6,100,nothing)


kern = np.zeros((1,1))
fimg = np.zeros_like(img)
gray_image = np.zeros_like(img)
thresh = np.zeros_like(img)
dilated = np.zeros_like(img)
imgLine = np.copy(img)

while(1):
    # cv.imshow('fimg',fimg)

    cv.imshow('tresh', dilated)

    cv.imshow('lines', imgLine)
    # cv.imshow('kern', kern)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    ksize = cv.getTrackbarPos('ksize','image')
    sigma = cv.getTrackbarPos('Sigma','image')
    theta = cv.getTrackbarPos('Theta','image')
    f = cv.getTrackbarPos('f','image')

    kern = np.real(gabor_kernel(1/f, theta=np.deg2rad(theta), sigma_x=sigma/10, sigma_y=sigma/10, n_stds=ksize))
    # plt.imshow(kern, cmap='gray')
    # kern = kern[:,:,None]

    # plt.pause(0.1)
    # fimg = ndi.convolve(img, kern, mode='wrap')
    fimg = cv.filter2D(img, cv.CV_8UC3, kern)
    accum = np.zeros_like(img)
    np.maximum(accum, fimg, accum)

    gray_image = cv.cvtColor(accum, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray_image, 7, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, np.ones((3, 3)))

    # lines = cv.HoughLinesP(dilated, rho=1, theta=np.pi / 90, threshold=10, minLineLength=50, maxLineGap=1)
    # imgLine = np.copy(img)
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv.line(imgLine, (x1, y1), (x2, y2), (0, 0, 255), 3)
    lines = cv.HoughLines(dilated, rho=1, theta=np.pi/45, threshold=250)
    imgLine = np.copy(img)
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))


            cv.line(imgLine,(x1,y1),(x2,y2),(0,0,255),4)