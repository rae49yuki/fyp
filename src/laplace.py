import cv2
import numpy as np

kernel_size = 3
scale = 1
delta = 0
ddepth = cv2.CV_16S

img = cv2.imread('51af9ccd43b8b90711.jpg')
img = cv2.GaussianBlur(img, (3,3), 0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_lap = cv2.Laplacian(gray, ddepth = ddepth, ksize = kernel_size, scale = scale, delta = delta)
dst = cv2.convertScaleAbs(gray_lap)

# cv2.imwrite('laplacian1.jpg', gray_lap)
cv2.imwrite('laplacian.jpg', dst)
# ret, thresh2 = cv2.threshold(gray_lap,64,255,cv2.THRESH_BINARY)
# cv2.imwrite('threshold.jpg', thresh2)