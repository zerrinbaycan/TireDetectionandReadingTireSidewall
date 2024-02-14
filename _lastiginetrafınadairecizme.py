#applyTireFlat test için yapıldı. linear polar ve warpolar metodları ile test edildi. warpolar metodu lastikte daha düz bir görüntü çıkarıyor.
#bu dosyayı çağırdığım bi yer yok
import cv2
import numpy as np

img = cv2.imread('C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop2\\6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.circle(img, (int(max(img.shape[0],img.shape[1])/2),int(max(img.shape[0],img.shape[1])/2)), int(max(img.shape[0],img.shape[1])/2), (255,255,255) , 8)
#cv2.circle(img, (int(img.shape[0]/2),int(img.shape[1]/2)), int(img.shape[1]/2), (255,255,255) , 8)

cv2.imshow("LAsitin dışına daire çizilmesi",img)
cv2.waitKey(0)


def linearPolarfunc(img):

    image = img.astype(np.float32)
    polar_image = cv2.linearPolar(src = image, center = (max(img.shape[0],img.shape[1])/2, max(img.shape[0],img.shape[1])/2), maxRadius = int(max(img.shape[0],img.shape[1])), flags = cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)

    cv2.imshow("",polar_image)
    cv2.waitKey(0)

    rotate_img = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow("rotate",rotate_img)
    cv2.waitKey(0)
    cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\flattire\\linearPolarlastikline6.jpg",rotate_img)


def warpPolarfunc(img):

    warped_img = cv2.warpPolar(img, (0,0), (max(img.shape[0],img.shape[1])  // 2, max(img.shape[0],img.shape[1]) // 2), max(img.shape[0],img.shape[1]), cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR) 
    cv2.imshow("warped_img",warped_img)
    cv2.waitKey(0)

    warped2_img = cv2.rotate(warped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow("warped_img",warped2_img)
    cv2.waitKey(0)
    cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\flattire\\warpPolarlinearPolarlastikline6.jpg",warped2_img)

    
linearPolarfunc(img)
warpPolarfunc(img)