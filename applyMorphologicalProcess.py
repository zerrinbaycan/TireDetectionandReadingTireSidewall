import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt
import applyOCR as ao

# (Contrast Limited Adaptive Histogram Equalization)   kontrast artırma ve görüntü iyileştirme amacıyla kullanılır.adaptif bir şekilde kontrastı artırmaktır. Bu yöntem, görüntüdeki farklı bölgelerin farklı kontrast düzeylerine sahip olabileceği durumları ele alır. Özellikle, bir görüntüde belirli bir bölgede kontrast arttırılırken, bu artışın komşu bölgelere olumsuz bir etki yapmamasını sağlar.
#histogram eşitleme genellikle genel kontrast iyileştirmesi için kullanılırken, gama düzeltme daha özelleştirilebilir parlaklık ve kontrast ayarı için tercih edilebilir. Laplacian filtresi ise özellikle kenar tespiti ve detayları belirginleştirmek için uygundur
def ClacheHistogram(img,clipLimit=4.0,tileGridSize=(8,8)):
    try:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        equalized = clahe.apply(img)

        return equalized
    
    except Exception as e:
        print(e)
        print("Hata")

def Histequalize(img):
    try:
        equ = cv2.equalizeHist(img)
        res = np.hstack((img,equ)) #stacking images side-by-side
        return equ
        
    except Exception as e:
        print(e)
        print("Hata")

#Bu metod applyTireFlat iççerisinden çağrılıyor.
def applyMorphologicalProcess(img,detectfile_dir,dosya_adi):
    try:
        #flattire klasöründeki lineer hale getirdiğimiz lastikleri okuyup işlemleri yapıyoruz
        mainfile_tire_dir = os.path.join(detectfile_dir, 'morphologicalProcesses')        
        if not os.path.exists(mainfile_tire_dir):
            os.mkdir(mainfile_tire_dir)

        Histequalize_dir = os.path.join(mainfile_tire_dir, 'Histequalize')        
        if not os.path.exists(Histequalize_dir):
            os.mkdir(Histequalize_dir)

        #burda ocr'a gönderilecek resim hazırlıyoruz
        img = cv2.equalizeHist(img)
        Histequalize_dir = os.path.join(Histequalize_dir, dosya_adi)
        cv2.imwrite(Histequalize_dir,img)

        Clachetire_dir = os.path.join(mainfile_tire_dir, 'ClacheHistogram')        
        if not os.path.exists(Clachetire_dir):
            os.mkdir(Clachetire_dir)

        img = ClacheHistogram(img,2.0,(8,8))
        Clachetire_dir = os.path.join(Clachetire_dir, dosya_adi)
        cv2.imwrite(Clachetire_dir,img)
        
        Histequalize_dir = os.path.join(mainfile_tire_dir, 'Histequalize2')        
        if not os.path.exists(Histequalize_dir):
            os.mkdir(Histequalize_dir)

        img2 = cv2.equalizeHist(img) #Kenar bulma sonrası histogram eşitleme yaptığımızda ocr sonuç bulmasında doğruluk artarmı kontrol etmek için ekledim
        Histequalize_dir = os.path.join(Histequalize_dir, dosya_adi)
        cv2.imwrite(Histequalize_dir,img2)

        ao.ApplyOcr(img,detectfile_dir,dosya_adi)
        
        ao.ApplyOcr(img2,detectfile_dir,("_" + dosya_adi))

        return img
        
            
    except Exception as e:
        print("Morphological Process Hata")

######################### Test için aşağıdaki kodlar kullanılabilir #####################################


#Bir dosya yolundaki resimler için morfolojik işlem metodunu çağıracaksak
"""
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\flattire"
for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        path = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop"
        image = applyMorphologicalProcess(img,path,dosya_adi)  
"""

#Tek bir resim için Ocr çalıştıracaksak
"""
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\1.jpg"
img = cv2.imread(yol,0)
ao.ApplyOcr(img,"C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin","1.jpg")
"""


#Bir dosya yolundaki resimler için Ocr çalıştıracaksak
"""
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3"
for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        path = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\zerrin\\3"
        ao.ApplyOcr(img,path,dosya_adi) 
"""