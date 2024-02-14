import cv2 
import numpy as np

#resim1de başarılı değil ama 2 ve 3de başarılı
img = cv2.imread('C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop2\\1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=int(max(img.shape[0],img.shape[1])/2),   minRadius = 0, maxRadius=int(max(img.shape[0],img.shape[1])) )

canvas = img.copy()
canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
## (3) Get the mean of centers and do offset
circles = np.int0(np.array(circles))
x,y,r = 0,0,0
for ptx,pty, radius in circles[0]:
    cv2.circle(canvas, (ptx,pty), radius, (0,255,0), 1, 16)
    x += ptx
    y += pty
    r += radius

cnt = len(circles[0])
x = x//cnt
y = y//cnt
r = r//cnt
x+=5
y-=7

"""
## (4) Draw the labels in red
for r in range(100, r, 20):
    cv2.circle(canvas, (x,y), r, (0, 0, 255), 3, cv2.LINE_AA)
cv2.circle(canvas, (x,y), 3, (0,0,255), -1)
"""

cv2.imshow("",canvas)
cv2.waitKey(0)