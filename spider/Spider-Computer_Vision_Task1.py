import cv2
import numpy as np

#Initialising callback function of trackbar
def nothing(x):
    pass

#Contour detection after smoothing the image
img_location = input('Enter image location:\n')
img = cv2.imread(img_location,-1)
blur = cv2.blur(img,(3,3))
im_gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(im_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_,contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#Creating window
cv2.namedWindow('Contours',cv2.WINDOW_NORMAL)

#Creating Trackbar
cv2.createTrackbar('Trackbar','Contours',0,len(contours)-1,nothing)

#Creating dictionary of perimeter of the contours
contoursDict,indice = {},0
for cnt in contours:
    perimeter = cv2.arcLength(cnt,True)
    contoursDict[indice] = perimeter
    indice += 1

#Making a contour having contours in ascending order
acs_contours = []
for item in sorted(contoursDict.items(),key = lambda x: x[1],reverse = False):
    acs_contours.append(contours[item[0]])

#Final loop
while True:

    #getting Trackbar position
    position = cv2.getTrackbarPos('Trackbar','Contours')

    #For making the image contour free when position changes
    blur = cv2.blur(img,(3,3))

    #Drawing contours of Red color with 3 pixel thickness 
    cv2.drawContours(blur,acs_contours,position,(0,0,255),3)
    cv2.imshow('Contours',blur)

    #Press 'q' to quit the program
    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()
