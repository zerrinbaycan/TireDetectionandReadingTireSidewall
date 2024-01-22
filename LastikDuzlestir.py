import cv2

def ApplyOcr(img):
    
    return

def ApplyMorphologicalProcess(img):
    
    cv2.imshow("",img)
    cv2.waitKey(0)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    w,h = gray.shape
    
    

def LastigiDuzlestir(img):
    
    # C:\ZerrinGit\TireSidewall\11_warpolarilelastigidüzlestirme.py içinde test ederek yaptım.
    cv2.imshow("",img)
    cv2.waitKey(0)

    w,h = img.shape    
    
    # warpPolar fonksiyonunu kullanarak görüntüyü dönüştürme
    t = (w // 2, h // 2)
    radius = w  # Dönüşümün merkezinden maksimum uzaklık
    warped_img = cv2.warpPolar(img, (0,0), (w  // 2, h // 2), radius, cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
    cv2.imshow('Dönüştürülmüş Görüntü 1', warped_img)

    warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow("rotate",warped_img)
    cv2.waitKey(0)

    cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\testresim6.jpg",warped_img)
    cv2.destroyAllWindows()

    ApplyOcr(img)