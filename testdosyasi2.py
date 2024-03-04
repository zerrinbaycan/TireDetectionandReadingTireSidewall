
######## Canny + median + dilate +erode işlemleri ile
"""
import cv2
import numpy as np

image = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\flattire\\20.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny1\\1.jpg",image)

#threshold2 küçüldükçe yazılar daha da beyazlaştı.büyülttükçe yazılar üstünde siyah gölgeler arttı.
# threshold1 küçülttükçe siyah zeminde beyazlamaya başladı o yüzden küçültmedim.
# Canny kenar tespiti
edges = cv2.Canny(image, 50, 150)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny\\Canny1\\2.jpg",edges)

# Kenarları genişletmek için  dilate uyguladım
kernel = np.ones((3,3), np.uint8)
thick_edges = cv2.dilate(edges, kernel, iterations=1)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny1\\3.jpg",thick_edges)

medianblur = cv2.medianBlur(thick_edges, 5)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny1\\4.jpg",medianblur)

image_temp = cv2.bitwise_not(thick_edges)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny1\\5.jpg",image_temp)

# Yazılardaki siyahları belirginleştirmek için erode işlemi uyguladım.
kernel = np.ones((3,3), np.uint8)
thick_edges = cv2.erode(image_temp, kernel, iterations=1)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny1\\6.jpg",thick_edges)
"""

######## Canny + histogram eşitleme + median + dilate +erode işlemleri ile
""" 
import cv2
import numpy as np

image = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\flattire\\resim20.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny3\\1.jpg",image)


image = cv2.equalizeHist(image)

#blurred = cv2.GaussianBlur(image, (0,0), 11)
#image = cv2.addWeighted(image, 2, blurred, -1, 0)

cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny3\\1_1.jpg",image)

# Canny kenar tespiti
edges = cv2.Canny(image, 50, 150)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny3\\2.jpg",edges)

# Kenarları genişletmek için  dilate uyguladım
kernel = np.ones((3,3), np.uint8)
thick_edges = cv2.dilate(edges, kernel, iterations=1)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny3\\3.jpg",thick_edges)

medianblur = cv2.medianBlur(thick_edges, 5)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny3\\4.jpg",medianblur)

image_temp = cv2.bitwise_not(thick_edges)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny3\\5.jpg",image_temp)

# Yazılardaki siyahları belirginleştirmek için erode işlemi uyguladım.
kernel = np.ones((3,3), np.uint8)
thick_edges = cv2.erode(image_temp, kernel, iterations=1)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny3\\6.jpg",thick_edges)
"""

######## Canny + Canny + adaptiveThreshold ile
"""
import cv2
import numpy as np

# Görüntüyü yükleyin
image = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\flattire\\resim20.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny2\\1.jpg",image)

# Canny kenar tespiti
edges = cv2.Canny(image, 100, 100)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny2\\2.jpg",edges)


# Kenarları siyah renge boyama
masked_image = np.copy(image)
masked_image[edges != 0] = 0  # Kenar piksellerini siyah renge boyar
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny2\\3.jpg",masked_image)

# Canny kenar tespiti
edges = cv2.Canny(masked_image, 100, 100)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny2\\4.jpg",edges)

# Kenar iç kısmındaki 3 pikseli de siyah renge boyama
kernel = np.ones((3,3), np.uint8)
inner_mask = cv2.dilate(edges, kernel, iterations=1)
masked_image[inner_mask != 0] = 0
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny2\\5.jpg",masked_image)

image_temp = cv2.adaptiveThreshold(masked_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#_, image_temp = cv2.threshold(masked_image, 120, 255,  cv2.ADAPTIVE_THRESH_MEAN_C )
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny2\\6.jpg",image_temp)

# Karanlık bölgeleri vurgulayan maske oluştur
image_temp = cv2.bitwise_not(image_temp)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\Canny\\Canny2\\7.jpg",image_temp)
"""