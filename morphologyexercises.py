#burda lastik üzerinde yapılacak işlemleri test etmek  için kod parçaları vardır.
#Testler burdan yapılabilir.
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

def SharpImage(image):
    blurred = cv2.GaussianBlur(image, (0,0), 5)
    sharp = cv2.addWeighted(image, 2, blurred, -1, 0)

    return sharp

#Lastiğe uygulanabilecek filtreleri denemeyi sağlıyor
"""
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop4\\zerrin"
for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)

    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        
        flattire_dir = os.path.join(yol, 'filtre')
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

        clache = SharpImage(img)
        temp_flattire_dir = os.path.join(flattire_dir, ( "SharpImage_" + dosya_adi)) 
        cv2.imwrite(temp_flattire_dir,clache)
  """       


#ClacheHistogram filtresi parametre değerlerini test etmeyi sağlıyor
"""
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop4\\zerrin"
for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)

    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        
        flattire_dir = os.path.join(yol, 'Clache')
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

#Lastiğe erosion,dilation morfolojik işlemlerini uygulama
"""
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

#Lastiğe uyguladığımız erode-dilate işlemleri kaydediyoruz
"""
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

#LAplacian filtresi. Resimler siyah oldu. işime yaramadı yada yanlış kullanmışda olabilirim
"""
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses"
clache_yol = os.path.join(yol, "ClacheHistogram")
treshold_tire_dir = os.path.join(yol, 'laplacian')
for dosya_adi in os.listdir(clache_yol):
    dosya_yolu = os.path.join(clache_yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        if not os.path.exists(treshold_tire_dir):
            os.mkdir(treshold_tire_dir)
        
        laplacian = cv2.Laplacian(img, cv2.CV_64F) 
        temp_flattire_dir = os.path.join(treshold_tire_dir,  dosya_adi)
        cv2.imwrite(temp_flattire_dir,laplacian)
"""

#Unsharp Masking uygulama. Bununla resimlerdeki yazıları daha netleştitmeye çalıştım.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\ClacheHistogram"

morphologicalProcesses_yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses"

for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)

        blurred = cv2.GaussianBlur(img, (0,0), 5)
        sharp = cv2.addWeighted(img, 2, blurred, -1, 15)
        
        file_name = os.path.join(morphologicalProcesses_yol, ("ClacheOrg_"+dosya_adi))
        cv2.imwrite(file_name,img)
        file_name = os.path.join(morphologicalProcesses_yol, ("Clache_15_"+dosya_adi))
        cv2.imwrite(file_name,sharp)
"""

# Frekans domaini örneği
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Resmi oku ve gri tonlamaya çevir
image_path = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop4\\zerrin\\151.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Fourier dönüşümü uygula
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Frekans domaininde bir filtre oluştur (yüksek geçişli filtre)
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2
radius = 30
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 0, thickness=-1)

# Yüksek geçişli filtreyi uygula
f_transform_shifted = f_transform_shifted * mask

# Ters Fourier dönüşümü uygula
f_ishift = np.fft.ifftshift(f_transform_shifted)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Giriş ve çıkış resimlerini göster
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('Sharpened Image in Frequency Domain')

cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop4\\zerrin\\151_fft.jpg",img_back)
cv2.imshow("1",image)
cv2.imshow("2",img_back)
cv2.waitKey(0)
plt.show()
"""

#Lastiğe 80-180 değerleri arasında  threshold işlemi uyguluyoruz
"""
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop4\\zerrin"
clache_yol =yol# os.path.join(yol, "Erode")
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

#Lastiğe uyguladığımız threshold işlemleri kaydediyoruz
"""
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
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop4\\zerrin"
clache_yol = yol# os.path.join(yol, "ClacheHistogram")
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


"""
image = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\flattire\\1.jpg", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (0,0), 5)
sharp = cv2.addWeighted(image, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1.jpg",image)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_1_Sharped.jpg",sharp)

img3 = ClacheHistogram(sharp,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_2_SharpedClache.jpg",sharp)


img = Histequalize(image)
blurred = cv2.GaussianBlur(img, (0,0), 5)
sharp = cv2.addWeighted(img, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_3_histSharped.jpg",sharp)

img3 = ClacheHistogram(sharp,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_4_histSharpedClache.jpg",sharp)


blurred = cv2.GaussianBlur(image, (0,0), 5)
sharp = cv2.addWeighted(img, 2, blurred, -1, 0)
sharp = Histequalize(sharp)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_5_histSharped.jpg",sharp)
img3 = ClacheHistogram(sharp,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_6_histSharpedClache.jpg",sharp)



img2 = Histequalize(image)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_7_Sharphist.jpg",img2)

img2 = ClacheHistogram(img2,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_8_SharphistClache.jpg",img2)



img2 = Histequalize(image)
img3 = ClacheHistogram(img2,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_9_histSharpedClache.jpg",sharp)
blurred = cv2.GaussianBlur(img3, (0,0), 5)
sharp = cv2.addWeighted(img3, 2, blurred, -1, 0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\1_10_histSharped.jpg",sharp)
"""


"""
# Resmi oku ve gri tonlamaya çevir
image = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\ClacheHistogram\\20.jpg", cv2.IMREAD_GRAYSCALE)

# Düzgünleştirme (blurring) filtresi oluştur
kernel = np.ones((5, 5), np.float32) / 25
# Konvolüsyon uygula
smoothed_image = cv2.filter2D(image, -1, kernel)

blurred = cv2.GaussianBlur(image, (0,0), 5)
sharp = cv2.addWeighted(image, 2, blurred, -1, 0)

# Resmi eşikleme (Thresholding) uygula
_, binary_image1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binary_image2 = cv2.threshold(smoothed_image, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binary_image3 = cv2.threshold(sharp, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\smoothed_image.jpg",smoothed_image)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\sharp.jpg",sharp)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\tresh.jpg",binary_image1)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\smoothed_imagetresh.jpg",binary_image2)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\sharptresh.jpg",binary_image3)
"""

"""
# Resmi oku ve gri tonlamaya çevir
image = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\ClacheHistogram\\20.jpg", cv2.IMREAD_GRAYSCALE)

# Histogram eşitleme uygula
equalized_image = cv2.equalizeHist(image)

# Adaptif eşikleme uygula
adaptive_threshold = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Morfolojik işlemler uygula (erozyon ve genişleme)
kernel = np.ones((3, 3), np.uint8)
morphology_result = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)

cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\11111.jpg",image)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\11112.jpg",adaptive_threshold)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\11113.jpg",morphology_result)


img3 = ClacheHistogram(morphology_result,2.0,(8,8))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\11113_ClacheHistogram.jpg",img3)

blurred = cv2.blur(img3,(3,3))
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\11113_blur.jpg",blurred)
"""


image = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop4\\zerrin\\17\\17.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop4\\zerrin\\17\\hsv.jpg",image)