import cv2
import numpy as np
from matplotlib import pyplot as plt

def ClacheHistogram(img,clipLimit,tileGridSize):
    try:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        equalized = clahe.apply(img)

        return equalized
    
    except Exception as e:
        print(e)
        print("Hata")

# Resmi oku ve gri tonlamaya çevir
image = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\flattire\\6.jpg", cv2.IMREAD_GRAYSCALE)

######adaptive_threshold + MORPH_OPEN + ClacheHistogram + medianBlur
"""
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\1\\1.jpg",image)
# Histogram eşitleme uygula
equalized_image = cv2.equalizeHist(image)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\1\\2_equalize_hist.jpg",equalized_image)

# Adaptif eşikleme uygula
adaptive_threshold = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\1\\3_adaptive_threshold.jpg",adaptive_threshold)

# Morfolojik işlemler uygula (erozyon ve genişleme)
kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\1\\4_morphology_result.jpg",morphology_result)


img3 = ClacheHistogram(morphology_result,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\1\\5_ClacheHistogram.jpg",img3)

blurred = cv2.medianBlur(img3,3)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\1\\6_blur.jpg",blurred)
"""


######## filter2D (convolution görüntü üzerinde özellik çıkarma, kenar tespiti veya diğer özel işlemleri gerçekleştirmek için kullanılır.)
"""
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\2\\1.jpg",image)
#image = cv2.equalizeHist(image)
alpha = 1.5
contrast_image = cv2.convertScaleAbs(image,alpha=alpha)#parlaklık ve kontrast ayarlaması yapmak için kullanılıyor
blurred = cv2.GaussianBlur(contrast_image, (0,0), 11)
contrast_image = cv2.addWeighted(contrast_image, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\2\\2.jpg",contrast_image)

kernel = np.array([[-1, -1 ,-1], [-1, 9, -1], [-1 ,-1 ,-1]])
sharpened_image = cv2.filter2D(image, -1,kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\2\\3.jpg",sharpened_image)
"""

######kenar bulma + treshold + erode-dilate
"""
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\1.jpg",image)
blurred = cv2.GaussianBlur(image, (0,0), 11)
sharp = cv2.addWeighted(image, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\2.jpg",sharp)

# Histogram eşitleme uygula
sharp = cv2.equalizeHist(sharp)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\3.jpg",sharp)


clache = ClacheHistogram(image,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\4.jpg",clache)
# Histogram eşitleme uygula                                                            
clache = cv2.equalizeHist(clache)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\5.jpg",clache)


blurred = cv2.GaussianBlur(clache, (0,0), 11)
sharp = cv2.addWeighted(clache, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\6.jpg",sharp)

# Adaptif eşikleme uygula
_, image_temp = cv2.threshold(sharp, 100, 255, cv2.THRESH_OTSU )
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\7.jpg",image_temp)
adaptive_threshold = cv2.adaptiveThreshold(image_temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\8.jpg",adaptive_threshold)

kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
erosion = cv2.erode(morphology_result,kernel,iterations=1)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\9.jpg",morphology_result)

kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(morphology_result, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\10.jpg",morphology_result)

#a = cv2.connectedComponentsWithStats()


clache = ClacheHistogram(sharp,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\11.jpg",clache)

# Adaptif eşikleme uygula
_, image_temp = cv2.threshold(sharp, 100, 255,  cv2.ADAPTIVE_THRESH_MEAN_C )
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\12.jpg",image_temp)
adaptive_threshold = cv2.adaptiveThreshold(image_temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\13.jpg",adaptive_threshold)

kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\14.jpg",morphology_result)

kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(morphology_result, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3\\15.jpg",morphology_result)
"""



###### kenar bulma + treshold +adaptive_threshold + erode-dilate
"""
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\4\\1.jpg",image)

blurred = cv2.GaussianBlur(image, (0,0), 11)
sharp = cv2.addWeighted(image, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\4\\2.jpg",sharp)

# Adaptif eşikleme uygula
_, image_temp = cv2.threshold(sharp, 100, 255,  cv2.THRESH_TRUNC)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\4\\3.jpg",image_temp)

adaptive_threshold = cv2.adaptiveThreshold(image_temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\4\\4.jpg",adaptive_threshold)

kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\4\\5.jpg",morphology_result)
"""


###### kenar bulma + histogram eşitleme + treshold +adaptive_threshold + erode-dilate
"""
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\5\\1.jpg",image)

blurred = cv2.GaussianBlur(image, (0,0), 11)
sharp = cv2.addWeighted(image, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\5\\2.jpg",sharp)

equalized_image = cv2.equalizeHist(sharp)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\5\\3.jpg",equalized_image)

# Adaptif eşikleme uygula
_, image_temp = cv2.threshold(equalized_image, 100, 255,  cv2.THRESH_TRUNC)
adaptive_threshold = cv2.adaptiveThreshold(image_temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\5\\4.jpg",adaptive_threshold)

kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\5\\5.jpg",morphology_result)
"""




#### histogram eşitleme + treshold + erode + dilate
"""
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\1.jpg",image)

# Histogram eşitleme uygula
equalized_image = cv2.equalizeHist(image)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\2_equalize_hist.jpg",equalized_image)


img3 = ClacheHistogram(equalized_image,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\3_ClacheHistogram.jpg",img3)

equalized_image = cv2.equalizeHist(img3)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\4_equalize_hist.jpg",equalized_image)

# Adaptif eşikleme uygula
adaptive_threshold = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\5_adaptive_threshold.jpg",adaptive_threshold)


kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\6_morphology_result.jpg",morphology_result)

kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\7_morphology_result.jpg",morphology_result)


equalized_image = cv2.equalizeHist(image)            
blurred = cv2.GaussianBlur(equalized_image, (5,5), 5)
image_temp = cv2.addWeighted(equalized_image, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\8_addWeighted.jpg",image_temp)


equalized_image = cv2.equalizeHist(image_temp)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\6\\9_equalize_hist.jpg",equalized_image)
"""