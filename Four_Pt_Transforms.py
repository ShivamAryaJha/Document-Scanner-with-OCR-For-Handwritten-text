#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import cv2
import imutils
from skimage.filters import threshold_local
import matplotlib.pyplot as plt


# In[9]:


cv2.__version__


# In[10]:


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped


# In[40]:


first =  cv2.imread("Desktop/PDFs/pdf5.jpg")
img =  cv2.imread("Desktop/PDFs/pdf5.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,gray = cv2.threshold(gray,210,255,cv2.THRESH_BINARY)
kernel = np.ones((7, 7))
gray = cv2.erode(gray, kernel, iterations=1) 
gray = cv2.dilate(gray, kernel, iterations=1)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# Blur the image
blur = cv2.GaussianBlur(thresh_inv,(1,1),0)

thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

mxarea = 0

mask = np.ones(img.shape[:2], dtype="uint8") * 255
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    if w*h>1000:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), -1)
        if w*h > mxarea:
            mxarea = w*h
            X, Y, W, H = x, y, w, h

res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))


#cut_out = cv2.rectangle(first, (X, Y), (X+W, Y+H), (0, 0, 0), 5)
cut_out = first[Y: Y+H, X: X+W]

#cv2.imshow("boxes",  imutils.resize(mask, height = 650))
cv2.imshow("final image",  imutils.resize(res_final, height = 650))
cv2.imshow("Cut_Out",  imutils.resize(cut_out, height = 650))

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[38]:


image = cv2.imread("Desktop/PDFs/pdf5.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

edged = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.GaussianBlur(edged, (5, 5), 0)
edged = cv2.Canny(edged, 75, 200)
#edged = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY,11,2)    #1
T = threshold_local(edged, 11, offset = 10, method = "gaussian")
edged = (edged > T).astype("uint8") * 255

pr = 2*(image.shape[0] + image.shape[1])
#cv2.imshow("Image", image)
#cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
        
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        print(peri)
        if  abs(cv2.arcLength(c, True) - pr) > 15 :
            screenCnt = approx
            break

cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 2) 
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[36]:


warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
cv2.imshow("Eh?", imutils.resize(warped, height = 650))
img = warped.copy()
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = cv2.Canny(warped, 75, 200)
ret, warped = cv2.threshold(warped, 127, 255, 0)

cnts = cv2.findContours(warped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(warped, cnts, -1, (0,0,0), 1)

digitCnts = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 15 and (h >= 30 and h <= 40):
        digitCnts.append(c)
#      cv2.rectangle(warped, x, y, (0, 255, 0), 2)

cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


# apply the four point transform to obtain a top-dow
# view of the original image
warped = cv2.imread("Desktop/example.jpg")
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#T = threshold_local(warped, 11, offset = 10, method = "gaussian")
#warped = (warped > T).astype("uint8") * 255


# In[14]:


warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
print(warped.shape)
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()

warp =  imutils.resize(warped, height = 650)


# In[ ]:


impo

