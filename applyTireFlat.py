import os
import cv2
import applyMorphologicalProcess as mp

#bu metod detect.py dosyasından çağrılıyor. 
#Resim içerisinde lastik olan alan tespit edilip lastik kırpılır ve merkezi bulunarak düz şerit haline getirilir.

def tireFlat(detectfile_dir):
    try:
        #kaydettiğimiz resim dosyaları içerisinde dönüp işlemleri yapıyoruz
        for dosya_adi in os.listdir(detectfile_dir):
            dosya_yolu = os.path.join(detectfile_dir, dosya_adi)

            if os.path.isfile(dosya_yolu):

                img = cv2.imread(dosya_yolu,0)   
                #img = cv2.resize(img, (416,416))        
                w,h = img.shape    
                
                # warpPolar fonksiyonunu kullanarak görüntüyü dönüştürme
                radius = w  # Dönüşümün merkezinden maksimum uzaklık
                warped_img = cv2.warpPolar(img, (0,0), (w  // 2, h // 2), radius, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)        
                rotate_img = cv2.rotate(warped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                flattire_dir = os.path.join(detectfile_dir, 'flattire')
                if not os.path.exists(flattire_dir):
                    os.mkdir(flattire_dir)

                flattire_dir = os.path.join(flattire_dir, dosya_adi) 
                cv2.imwrite(flattire_dir,rotate_img)               

                mp.applyMorphologicalProcess(rotate_img,detectfile_dir,dosya_adi)            
            
    except Exception as e:
        print("LastigiDuzlestir Hata")

#yolo ile tespit ettiğimiz sadece lastik olan alanı crop klasörüne kaydetmiştik.
#Aşağıdaki kod satırını açarak elle crop dosyasındaki resimlere düzleştirme + morfolojik işlem + ocr uygulanabilir
#tireFlat('data\\images\\detectimages\\crop')