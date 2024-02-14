#burda lastik üzerinde yapılacak işlemleri test etmek  için kod parçaları vardır.
import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt

#################histogram eşitleme genellikle genel kontrast iyileştirmesi için kullanılırken, gama düzeltme daha özelleştirilebilir parlaklık ve kontrast ayarı için tercih edilebilir. Laplacian filtresi ise özellikle kenar tespiti ve detayları belirginleştirmek için uygundur#################

# (Contrast Limited Adaptive Histogram Equalization)   kontrast artırma ve görüntü iyileştirme amacıyla kullanılır.adaptif bir şekilde kontrastı artırmaktır. Bu yöntem, görüntüdeki farklı bölgelerin farklı kontrast düzeylerine sahip olabileceği durumları ele alır. Özellikle, bir görüntüde belirli bir bölgede kontrast arttırılırken, bu artışın komşu bölgelere olumsuz bir etki yapmamasını sağlar.
def ClacheHistogram(img,clipLimit,tileGridSize):
    try:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        equalized = clahe.apply(img)

        return equalized
    
    except Exception as e:
        print(e)
        print("Hata")

#bir görüntüdeki kontrast değişimlerini vurgulamaya veya belirginleştirmeye yarar. Kullanım alanları; Kenar Tespiti, Görüntü İyileştirme,Morfolojik Operasyonlarda,Görüntü Sıkıştırma. Gürültüye duyarlı olduğu ve gürültüyü artırabileceği göz önünde bulundurulmalıdır
def LaplaceFilter(img):

    # Laplace filtresi uygula
    laplacian = cv2.Laplacian(img, cv2.CV_32F)
    return laplacian

#Bir görüntünün parlaklık ve kontrastını ayarlamak için kullanılan bir tekniktir. Gama filtresinin temel işlevleri; Parlaklık Ayarı, Kontrast Ayarı, Renk Doyma Ayarı, Renk Düzeltme, Görüntü İyileştirme.
def gammaFilter(img):
    # Gama düzeltme parametresi
    gamma = 1.5

    # Gama düzeltme uygula
    gamma_corrected = np.power(img/float(np.max(img)), gamma)
    return gamma_corrected

#Bir görüntünün kontrastını artırmak ve görüntüdeki detayları daha belirgin hale getirmek amacıyla kullanılır.
def Histequalize(img):
    try:
        equ = cv2.equalizeHist(img)
        res = np.hstack((img,equ)) #stacking images side-by-side
        return equ
        """
        ret, thresh = cv2.threshold(equ, 100, 255, cv2.THRESH_BINARY)

        cv2.imshow("res2",thresh)
        cv2.waitKey(0)

        kernel = np.ones((3,3),np.uint8)

        thresh = cv2.dilate(thresh,kernel,iterations=1)
        cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\morfolojik\\testresim3_equalizeHist_tresh.jpg",thresh)
        """
        
    except Exception as e:
        print(e)
        print("Hata")


def ApplyHistogramProcess(img):    
    try:
        #convert to NumPy array
        img_array = np.asarray(img)

        ######################################
        # PERFORM HISTOGRAM EQUALIZATION
        ######################################

        """
        STEP 1: Normalized cumulative histogram
        """
        #flatten image array and calculate histogram via binning
        histogram_array = np.bincount(img_array.flatten(), minlength=256)

        #normalize
        num_pixels = np.sum(histogram_array)
        histogram_array = histogram_array/num_pixels

        #normalized cumulative histogram
        chistogram_array = np.cumsum(histogram_array)

        """
        STEP 2: Pixel mapping lookup table
        """
        transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

        """
        STEP 3: Transformation
        """
        # flatten image array into 1D list
        img_list = list(img_array.flatten())

        # transform pixel values to equalize
        eq_img_list = [transform_map[p] for p in img_list]

        # reshape and write back into img_array
        eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

        return eq_img_array

    except :
        print("ApplyHistogramProcess hata")

"""
#Lastiğe uygulanabilecek filtreleri denemeyi sağlıyor
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\tresholdtire"
for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)

    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        
        flattire_dir = os.path.join(yol, 'treshold')
        if not os.path.exists(flattire_dir):
            os.mkdir(flattire_dir)

        clache = ClacheHistogram(img,4.0,(16,16))
        temp_flattire_dir = os.path.join(flattire_dir, ( "clache_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,clache)    

        clache = LaplaceFilter(img)
        temp_flattire_dir = os.path.join(flattire_dir, ( "laplace_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,clache) 

        clache = gammaFilter(img)
        temp_flattire_dir = os.path.join(flattire_dir, ( "gamma_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,clache) 

        clache = Histequalize(img)
        temp_flattire_dir = os.path.join(flattire_dir, ( "histequal_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,clache) 

        clache = ApplyHistogramProcess(img)
        temp_flattire_dir = os.path.join(flattire_dir, ( "ApplyHistogram_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,clache) 
"""

"""
#ClacheHistogram filtresi parametre değerlerini test etmeyi sağlıyor
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\tresholdtire"
for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)

    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        
        flattire_dir = os.path.join(yol, 'treshold')
        if not os.path.exists(flattire_dir):
            os.mkdir(flattire_dir)

        clahe = ClacheHistogram(img,2.0, (3,3))
        temp_flattire_dir = os.path.join(flattire_dir, ( "2_3x3_" + dosya_adi))    
        cv2.imwrite(temp_flattire_dir,clahe)

        clahe = ClacheHistogram(img,clipLimit=2.0, tileGridSize=(8,8)) 
        temp_flattire_dir = os.path.join(flattire_dir, ( "2_8x8_" + dosya_adi))    
        cv2.imwrite(temp_flattire_dir,clahe)

        clahe = ClacheHistogram(img,clipLimit=2.0, tileGridSize=(16,16))  
        temp_flattire_dir = os.path.join(flattire_dir, ( "2_16x16_" + dosya_adi))    
        cv2.imwrite(temp_flattire_dir,clahe)

        clahe = ClacheHistogram(img,clipLimit=4.0, tileGridSize=(3,3))   
        temp_flattire_dir = os.path.join(flattire_dir, ( "4_3x3_" + dosya_adi))    
        cv2.imwrite(temp_flattire_dir,clahe)


        clahe = ClacheHistogram(img,clipLimit=4.0, tileGridSize=(8,8))
        temp_flattire_dir = os.path.join(flattire_dir, ( "4_8x8_" + dosya_adi))    
        cv2.imwrite(temp_flattire_dir,clahe)

        clahe = ClacheHistogram(img,clipLimit=4.0, tileGridSize=(16,16))  
        temp_flattire_dir = os.path.join(flattire_dir, ( "4_16x16_" + dosya_adi))    
        cv2.imwrite(temp_flattire_dir,clahe)
        
        clahe = ClacheHistogram(img,clipLimit=3.0, tileGridSize=(16,16))  
        temp_flattire_dir = os.path.join(flattire_dir, ( "3_16x16_" + dosya_adi))    
        cv2.imwrite(temp_flattire_dir,clahe)
"""

"""
#Lastiğe erosion,dilation morfolojik işlemlerini uygulama
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\tresholdtire\\treshold"
for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        kernel = np.ones((3,3),np.uint8)

        erosion = cv2.erode(img,kernel,iterations=1)
        dilation = cv2.dilate(img,kernel,iterations=1)
        dilation2 = cv2.dilate(erosion,kernel,iterations=1)

        #Öce Erosion, sonra Dilation işlemlerini yapıyor
        opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel )
        #Öce Dilation, sonra Erosion işlemlerini yapıyor
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel )
        #Resimde Dilation - Eroison işlemi
        gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel )
        #Orjinal resşmin openingden farkı
        tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel )
        #Orjinal resşmin closingden farkı
        blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel )
        
        temp_flattire_dir = os.path.join(yol, ( "M_Original_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,img)  
        
        temp_flattire_dir = os.path.join(yol, ( "M_erosion_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,erosion)  

        temp_flattire_dir = os.path.join(yol, ( "M_dilation_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,dilation)  

        temp_flattire_dir = os.path.join(yol, ( "M_dilation2_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,dilation2)  

        temp_flattire_dir = os.path.join(yol, ( "M_opening_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,opening)  

        temp_flattire_dir = os.path.join(yol, ( "M_closing_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,closing)  

        temp_flattire_dir = os.path.join(yol, ( "M_gradient_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,gradient)  

        temp_flattire_dir = os.path.join(yol, ( "M_tophat_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,tophat)  

        temp_flattire_dir = os.path.join(yol, ( "M_blackhat_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,blackhat)  
"""

"""
#Lastiğe uyguladığımız erode-dilate işlemleri kaydediyoruz
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses"
clache_yol = os.path.join(yol, "ClacheHistogram")
erode_tire_dir = os.path.join(yol, 'clache_erode_dilate')
for dosya_adi in os.listdir(clache_yol):
    dosya_yolu = os.path.join(clache_yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        kernel = np.ones((3,3),np.uint8)

        erosion = cv2.erode(img,kernel,iterations=1)
        dilation = cv2.dilate(img,kernel,iterations=1)       
        
        if not os.path.exists(erode_tire_dir):
            os.mkdir(erode_tire_dir)

        temp_flattire_dir = os.path.join(erode_tire_dir, ( "erosion_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,erosion)  

        temp_flattire_dir = os.path.join(erode_tire_dir, ( "dilation_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,dilation)  
"""

"""
#Lastiğe 80-180 değerleri arasında  threshold işlemi uyguluyoruz
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses"
clache_yol = os.path.join(yol, "ClacheHistogram")
treshold_tire_dir = os.path.join(yol, 'treshold')
for dosya_adi in os.listdir(clache_yol):
    dosya_yolu = os.path.join(clache_yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        if not os.path.exists(treshold_tire_dir):
            os.mkdir(treshold_tire_dir)

        ret, thresh = cv2.threshold(img, 80, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_80_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh)  

        ret, thresh = cv2.threshold(img, 90, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_90_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 100, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_100_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 110, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_110_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 120, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_120_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 130, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_130_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 150, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_150_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 180, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_180_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 
        ret, thresh = cv2.threshold(img, 80, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_80_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh)  

        ret, thresh = cv2.threshold(img, 90, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_90_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 100, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_100_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 110, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_110_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 120, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_120_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 130, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_130_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 150, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_150_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 180, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_180_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh)
"""

"""

#Lastiğe uyguladığımız threshold işlemleri kaydediyoruz
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses"
clache_yol = os.path.join(yol, "ClacheHistogram")
treshold_tire_dir = os.path.join(yol, 'treshold')
for dosya_adi in os.listdir(clache_yol):
    dosya_yolu = os.path.join(clache_yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        if not os.path.exists(treshold_tire_dir):
            os.mkdir(treshold_tire_dir)

        ret, thresh = cv2.threshold(img, 80, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_80_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh)  

        ret, thresh = cv2.threshold(img, 90, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_90_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 100, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_100_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 110, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_110_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 120, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_120_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 130, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_130_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 150, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_150_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 180, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_180_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 
        ret, thresh = cv2.threshold(img, 80, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_80_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh)  

        ret, thresh = cv2.threshold(img, 90, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_90_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 100, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_100_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 110, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_110_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 120, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_120_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 130, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_130_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 150, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_150_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh) 

        ret, thresh = cv2.threshold(img, 180, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_180_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,thresh)
"""

#Lastiğe uyguladığımız adaptiveThreshold işlemleri kaydediyoruz
"""
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses"
clache_yol = os.path.join(yol, "ClacheHistogram")
treshold_tire_dir = os.path.join(yol, 'AdaptiveTreshold')
for dosya_adi in os.listdir(clache_yol):
    dosya_yolu = os.path.join(clache_yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        if not os.path.exists(treshold_tire_dir):
            os.mkdir(treshold_tire_dir)
        
        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3,10)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_3_10_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)  
        
        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3,20)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_3_20_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,10)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_5_10_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)  

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,20)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_5_20_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,10)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_7_10_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)  

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,20)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "GAUSSIAN_7_20_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,10)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_3_10_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)  

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,20)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_3_20_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5,10)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_5_10_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)  

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5,20)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_5_20_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7,10)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_7_10_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)  

        imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7,20)
        temp_flattire_dir = os.path.join(treshold_tire_dir, ( "MEAN_7_20_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,imgt)
"""