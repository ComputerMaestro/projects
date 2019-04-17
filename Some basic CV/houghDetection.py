import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.resize(cv2.imread('test8.jpg', cv2.IMREAD_COLOR), (700, 700))
img = cv2.resize(cv2.imread('test8.jpg', cv2.IMREAD_GRAYSCALE), (700, 700))

img = cv2.medianBlur(img, 9)
img = cv2.medianBlur(img, 9)
image = cv2.bitwise_not(img)
height, width = image.shape

edges = cv2.Canny(img, 200, 200, 5, 5, True)
contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(cnt) for cnt in contours]
lines = cv2.HoughLines(edges,1,np.pi/180,150)
linesim = image.copy()
linesim = cv2.cvtColor(linesim, cv2.COLOR_GRAY2BGR)
for rho, theta in lines.reshape((-1, 2)):
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))
      linesim = cv2.line(linesim, (x1, y1), (x2, y2), (0, 0, 255), 1)

##angle = 0.0
##nlines = 0
##up = 0
##down = 0
##for x1, y1, x2, y2 in lines.reshape(len(lines), 4):
##      x = (np.arctan((y2 - y1)/(x2 - x1))*180)/np.pi
##      if x > 5 : up += 1
##      if x < -5 : down += 1
##      if x > -5: continue
##      linesim = cv2.line(linesim, (x1, y1), (x2, y2), (255, 255, 255), 4)
##      print(x1, y1, x2, y2)
##      angle += x
##      nlines += 1
##      print(x)
##      print(up, down)
##      print(angle/nlines, angle, nlines, sep='\n')
##      print(len(contours))
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
##for cnt in contours:
##      cv2.drawContours(img, contours, -1, (255, 0, 0), 4)

cv2.drawContours(img, contours, areas.index(max(areas)), (0, 0, 255), 4)
cv2.imshow('canny', edges)
cv2.imshow('original', img1)
cv2.imshow('Hought lines', linesim)

cv2.waitKey(0)
cv2.destroyAllWindows()

