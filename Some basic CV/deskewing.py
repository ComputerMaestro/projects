import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Capture3.png', cv2.IMREAD_GRAYSCALE)

def compute_skew(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape

    edges = cv2.Canny(image, 150, 200, 3, 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=width / 2.0, maxLineGap=20)
    linesim = image.copy()
    for line in lines:
          linesim = cv2.line(linesim, (line[0, 0], line[0, 1]), (line[0, 2], line[0, 3]), (0, 255, 0), 1)
 
    angle = 0.0
    nlines = 0
    up = 0
    down = 0
    for x1, y1, x2, y2 in lines.reshape(len(lines), 4):
        x = (np.arctan((y2 - y1)/(x2 - x1))*180)/np.pi
        if x > 5 : up += 1
        if x < -5 : down += 1
        if x > -5: continue
        linesim = cv2.line(linesim, (x1, y1), (x2, y2), (255, 255, 255), 2)
        print(x1, y1, x2, y2)
        angle += x
        nlines += 1
        print(x)
        print(up, down)
    print(angle/nlines, angle, nlines, sep='\n')
    cv2.imshow('linesim', linesim) 
    return angle / nlines


def deskew(image, angle):
    image = cv2.bitwise_not(image)
    non_zero_pixels = cv2.findNonZero(image)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = image.shape
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    return cv2.getRectSubPix(rotated, (cols, rows), center)


deskewed_image = deskew(img.copy(), compute_skew(img))
cv2.imshow('original', img)
cv2.imshow('deskewed', deskewed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
