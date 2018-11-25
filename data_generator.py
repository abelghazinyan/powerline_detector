import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def contour_corrector(contours):
    contour_mean = []
    for x, y in pairwise(contours):
        contour = []
        for x_i, y_i in zip(x, y):
            contour.append([int((x_i[0][0] + y_i[0][0])/2), int((x_i[0][1] + y_i[0][1])/2)])
        contour = contour[0::100].copy()
        contour_mean.append(contour)
    return contour_mean

img = cv.imread("data/original/img/1.JPG")
img = np.array(img)
mask = cv.imread("data/original/mask/1.png",0)
blank = np.zeros((4000, 6000),np.uint8)

def rotate_img(img, angle, center):
    res = img
    r = cv.getRotationMatrix2D(center, angle, 1)
    res = cv.warpAffine(img, r, (4000, 6000));
    return res

edges = cv.Canny(mask,100,200)
cv.imwrite("edges.jpg", edges)

im2, contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(blank, contours, -1, (255,255,255), 4)

new_contours = contour_corrector(contours)

i = 0
for con in new_contours:
    for dots in con:
        cv.circle(blank, tuple(dots), 1, (255,255,255), cv.LINE_4)
        if (dots[0] - 20 > 0) and (dots[1] + 20 < blank.shape[0]) and (dots[0] + 20 < blank.shape[1]) and (dots[1] - 20 > 0):
            # cv.rectangle(img, (dots[0] - 20, dots[1] + 20), (dots[0] + 20, dots[1] - 20), (255,255,255), 10, -1)
            i+=1
            rect_img = np.array((40,40, 3), np.float)
            rect_mask = np.array((40,40, 1), np.float)
            rect_img = cv.getRectSubPix(img, (40, 40), tuple(dots))
            rect_mask = cv.getRectSubPix(mask, (40, 40), tuple(dots))
            cv.imwrite('data/batch/img/{}.jpg'.format(i),rect_img)
            cv.imwrite('data/batch/mask/{}.jpg'.format(i), rect_mask)

            for alpha in range(10, 360, 10):
                i+=1
                rotated_img = rotate_img(img,alpha, tuple(dots))
                rotated_mask = rotate_img(mask,alpha, tuple(dots))
                rect_img = cv.getRectSubPix(rotated_img, (40, 40), tuple(dots))
                rect_mask = cv.getRectSubPix(rotated_mask, (40, 40), tuple(dots))
                cv.imwrite('data/batch/img/{}.jpg'.format(i), rect_img)
                cv.imwrite('data/batch/mask/{}.jpg'.format(i), rect_mask)

print(len(new_contours[0]))
