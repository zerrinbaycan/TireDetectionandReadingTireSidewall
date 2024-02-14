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

        #burda ocr'a gönderilecek resim hazırlıyoruz
        img = ClacheHistogram(img,4.0,(16,16))
        Clachetire_dir = os.path.join(Clachetire_dir, dosya_adi)
        cv2.imwrite(Clachetire_dir,img)

        return img
        #ao.ApplyOcr()
            
    except Exception as e:
        print("Morphological Process Hata")
###################################################################################################################################################

yol = "C:\\Users\\Zerrin Baycan\\Desktop\\testresim"
for dosya_adi in os.listdir(yol):
    dosya_yolu = os.path.join(yol, dosya_adi)
    
    if os.path.isfile(dosya_yolu):

        img = cv2.imread(dosya_yolu,0)
        image = applyMorphologicalProcess(img,yol,dosya_adi)  

        gray10 = cv2.threshold(image,100,255,cv2.THRESH_OTSU)[1]  
        mainfile_tire_dir = os.path.join(yol, 'morphologicalProcesses') 

        Threshold_dir = os.path.join(mainfile_tire_dir, 'Threshold')        
        if not os.path.exists(Threshold_dir):
            os.mkdir(Threshold_dir)

        Threshold_dir = os.path.join(Threshold_dir, dosya_adi)
        cv2.imwrite(Threshold_dir,gray10)

        ao.ApplyOcr(img,dosya_adi)


