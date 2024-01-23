import cv2
import numpy as np

#bu metod detect.py dosyasından çağrılıyor. 
#Resim içerisinde lastik olan alan tespit edilip lastik kırpılır ve merkezi bulunarak düz şerit haline getirilir.
def LastigiDuzlestir(img):
    try:
        # C:\ZerrinGit\TireSidewall\11_warpolarilelastigidüzlestirme.py içinde test ederek yaptım.
        cv2.imshow("",img)
        cv2.waitKey(0)

        #img = cv2.resize(img, (416,416))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        w,h = img.shape    
        
        # warpPolar fonksiyonunu kullanarak görüntüyü dönüştürme
        radius = w  # Dönüşümün merkezinden maksimum uzaklık
        warped_img = cv2.warpPolar(img, (0,0), (w  // 2, h // 2), radius, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
        cv2.imshow('Dönüştürülmüş Görüntü 1', warped_img)

        warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("rotate",warped_img)
        cv2.waitKey(0)

        cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\lastikline1.jpg",warped_img)
        cv2.destroyAllWindows()

        #ApplyMorphologicalProcess(warped_img)

    except:
        print("LastigiDuzlestir Hata")









def ApplyOcr(img):
    try:
        return
    except:
        print("ApplyOcr Hata")











def ApplyHistogramProcess(img):    
    try:
        cv2.imshow("1",img)
#        cv2.waitKey(0)

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

        cv2.imshow("",eq_img_array)
        cv2.waitKey(0)

        cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\testresim7_histeq.jpg",eq_img_array)

        ApplyOcr(img)
    except :
        print("ApplyHistogramProcess hata")


def ApplyMorphologicalProcess(img):
    
    try:
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(img,kernel,iterations=1)
        dilation = cv2.dilate(img,kernel,iterations=1)

        cv2.imshow("erosion",erosion)
        cv2.waitKey(0)

        cv2.imshow("dilation",dilation)
        cv2.waitKey(0)

        erosion = cv2.erode(img,kernel,iterations=1)
        dilation = cv2.dilate(img,kernel,iterations=1)

        cv2.imshow("erosion",erosion)
        cv2.waitKey(0)

        cv2.imshow("dilation",dilation)
        cv2.waitKey(0)

        a = 1
    except:
        print("ApplyMorphologicalProcess hata")




def HistogramTest(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow("",gray)

        equ = cv2.equalizeHist(gray)
        res = np.hstack((gray,equ)) #stacking images side-by-side
        cv2.imshow("res",res)
        cv2.waitKey(0)

    except Exception as e:
        print(e)
        print("Hata")

def ClacheHistogramTest(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        cv2.imshow("",gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        cv2.imshow("res",equalized)
        cv2.waitKey(0)

        cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\testresim8_Clache.jpg",equalized)

    except Exception as e:
        print(e)
        print("Hata")

img = cv2.imread('C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\testresim6.jpg')
ClacheHistogramTest(img)
#ApplyMorphologicalProcess(img)