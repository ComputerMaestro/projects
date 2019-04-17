import cv2

image = cv2.imread('output4.jpg', 0)

ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
blur = cv2.medianBlur(thresh, 9)
contours, h = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
areas = [cv2.contourArea(cnt) for cnt in contours]

for cnt in contours:
      cv2.drawContours(image, contours, -1, (255, 0, 0), 4)
cv2.drawContours(image, contours, areas.index(max(areas)), (0, 0, 255), 4)
cv2.namedWindow('contours', cv2.WINDOW_AUTOSIZE)
image = cv2.resize(image, (900, 900))
blur = cv2.resize(blur, (900, 900))
cv2.imshow('contours', image)
cv2.imshow('blurred', blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
