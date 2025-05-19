import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:/Users/Ryan/Desktop/test_fig.png")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_new = remove_background(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray image', img_gray)
ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
# thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.imshow('Binary image', thresh)
# cv2.waitKey(0)

image_copy = img.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

areas = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i], False)
    areas.append(area)
    print("contour%d area:%d" % (i, area))

# ratio = areas[2]/areas[1]
# print("ratio of object: ", ratio*100, "%")
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)



