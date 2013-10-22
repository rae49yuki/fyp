import cv2
import numpy as np

lowThreshold = 65
ratio = 3
kernel_size = 3

img = cv2.imread('51af9ccd43b8b90711.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detected_edges = cv2.GaussianBlur(gray, (3,3), 0)
detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold*ratio, apertureSize = kernel_size)
#dst = cv2.bitwise_and(img, img, mask = detected_edges)
cv2.imwrite('canny.jpg', detected_edges)

ret, thresh2 = cv2.threshold(detected_edges, 128, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('threshold2.jpg', thresh2)
res = cv2.bitwise_and(gray, thresh2)
# cv2.imwrite('res.jpg', res)
res2 = cv2.bitwise_and(res, thresh2)
cv2.imwrite('res.jpg', res)

# def CannyThreshold(lowThreshold):
#     detected_edges = cv2.GaussianBlur(gray,(3,3),0)
#     detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
#     dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo',dst)

# lowThreshold = 0
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 3

# img = cv2.imread('How-to-Apply-Blush-on-Oval-Face.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('canny demo')

# cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)

# CannyThreshold(0)  # initialization
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()