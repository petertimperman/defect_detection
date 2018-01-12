import cv2
import numpy as np


filename ='board.jpg'
img = cv2.imread(filename,0)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = img 
gray = np.float32(img)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]
ret,img =cv2.threshold(img, .01*dst.max(), 255, cv2.THRESH_BINARY)
cv2.imshow('dst', dst)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
