import cv2 as cv
import numpy as np
import argparse
import os
from skimage.filters import gabor_kernel

parser = argparse.ArgumentParser()

parser.add_argument("--img",help="Image to process")
parser.add_argument("--dir",default="results",help="Directory to save results")
parser.add_argument("--theta_a",type=int,default="-90",help="Start angle of Gabor filter bank")
parser.add_argument("--theta_b",type=int,default="90",help="End angle of Gabor filter bank")
parser.add_argument("--inpaint",help="Remove powerlines from input",action="store_true")
parser.add_argument("--steps",help="Show steps of filtration",action="store_true")

args = parser.parse_args()

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

#Thresholding values for powerline color in HSV color space
low_H = 0
low_S = 45
low_V = 0

high_H = 90
high_S = 255
high_V = 255

#Get angle between two vectors with arcat
def angle(v1, v2, acute):
    if v1[0]-v2[0] == 0:
        return np.pi/2
    angle = np.arctan((v1[1]-v2[1])/(v1[0]-v2[0]))
    return np.rad2deg(angle)

def draw_lines(lines, mask):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 3)
    return mask

def average_angle(lines):
    avrg = 0.0
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                avrg += angle((x1,y1), (x2,y2), True)
        avrg = avrg / lines.shape[0]
    return avrg

img = cv.imread(args.img)#read the image

#Convert image to HSV color space and threshold
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

thresh_hsv = cv.inRange(img_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
thresh_hsv = 255 - thresh_hsv

if args.steps:
    cv.imwrite("{}/1_thresh_HSV.jpg".format(args.dir),thresh_hsv)

#Apply 
thresh = np.zeros_like(thresh_hsv)#store threshold

THETA_A = args.theta_a
THETA_B = args.theta_b

kernel = np.ones((1,1),np.uint8)#kernel for dilation

#Apply Gabor filter for theta with predefined filter arguments
for theta in range(THETA_A, THETA_B, 2):
    kern = np.real(gabor_kernel(1 / 12, theta=np.deg2rad(theta), sigma_x=35 / 6.3, sigma_y=35 / 6.3))
    fimg = cv.filter2D(img, cv.CV_8UC3, kern)
    accum = np.zeros_like(fimg)
    np.maximum(accum, fimg, accum)
    accum = cv.cvtColor(accum,cv.COLOR_BGR2GRAY)
    ret, thresh_tmp = cv.threshold(accum, 9, 255, cv.THRESH_BINARY) #threshold from gabor filter

    thresh_tmp = cv.dilate(thresh_tmp, kernel, iterations=3)

    thresh = cv.bitwise_or(thresh, thresh_tmp) # collecting overal lines

thresh = cv.bitwise_and(thresh_hsv,thresh) # hsv and gabor threshold bitwise and
thresh = cv.dilate(thresh, kernel, iterations=1)

if args.steps:
    cv.imwrite("{}/2_thres_GABOR.png".format(args.dir), thresh)#save gabor threshold after bitwise and with HSV threshold

#Find line segments with probabilistic Hough Transform
lines = cv.HoughLinesP(thresh, rho=0.2, theta=np.pi / 90, threshold=40, minLineLength=80, maxLineGap=3)
line_mask = np.zeros_like(thresh)#binary mask image of drawn line segments
line_mask = draw_lines(lines,line_mask)

if args.steps:
    cv.imwrite("{}/3_mask_hough.png".format(args.dir), line_mask)

angles = np.array([])#angles of line segments

if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            if angles.shape[0] == 0:
                angles = np.array([angle([x1,y1], [x2,y2], True)])
            else:
                angles = np.append(angles,[angle([x1,y1], [x2,y2], True)],axis=0)

ANGLE_THRESHOLD = 7
avrg_angle = average_angle(lines)

angle_label = np.zeros((angles.shape[0],1)).astype(int)
i = 0
for a in angles:
    if a > avrg_angle + ANGLE_THRESHOLD or a < avrg_angle - ANGLE_THRESHOLD:
        angle_label[i] = 1
    i += 1 
#Remove outlier segments
i = 0
for p in angle_label:
    if p == 1:
        lines = np.delete(lines, i, axis=0)
        i -= 1
    i+=1
    
#Make mask after removing outliers
line_mask = np.zeros_like(line_mask)
line_mask = draw_lines(lines, line_mask)

if args.steps:
    cv.imwrite("{}/4_mask_filtered.png".format(args.dir), line_mask)

#Find line segments with probabilistic Hough Transform
lines_over = cv.HoughLinesP(line_mask, rho=7, theta=np.pi / 90, threshold=200, minLineLength=10, maxLineGap=500)
line_mask_over = np.zeros_like(thresh)#binary mask image of drawn line segments
line_mask_over = draw_lines(lines_over,line_mask_over)

if args.steps:
    cv.imwrite("{}/5_mask_overline.png".format(args.dir), line_mask_over)

#label found lines and remove small ones

ret, labels = cv.connectedComponents(line_mask_over)
assert ret != 0,"NO lines"
# Map component labels to hue val
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
# cvt to BGR for display
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

if args.steps:
    cv.imwrite("{}/6_labeled.png".format(args.dir),labeled_img)

avrg_surface = 0
for i in range(1,ret):
    avrg_surface += labels[labels==i].shape[0]
avrg_surface /= ret - 1

SURFACE_THRESH = 0.5

for i in range(1,ret):
    if labels[labels==i].shape[0] < avrg_surface*SURFACE_THRESH:
        labeled_img[labels==i] = 0

if args.steps:
    cv.imwrite("{}/7_labeled_filtered.png".format(args.dir),labeled_img)

labeled_img = cv.cvtColor(labeled_img, cv.COLOR_BGR2GRAY)

ret, result = cv.threshold(labeled_img, 1, 255, cv.THRESH_BINARY)

cv.imwrite("{}/8_result.png".format(args.dir),result)

if args.inpaint:
    #Inpaint: remove detected lines from the image
    dst = cv.inpaint(img,result,3,cv.INPAINT_TELEA)
    cv.imwrite("{}/9_inpaint.png".format(args.dir), dst)
