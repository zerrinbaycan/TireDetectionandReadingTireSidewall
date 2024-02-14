#pytesseract ile yazıları tespit etmeye çalıştım ama başarılı olmadı. easyocr metinleri bulmada çok daha başarılı o yüzden easyocr ile devam edicem
import easyocr
import cv2
import numpy as np

def ApplyOcr(img,dosya_Adi):
    try:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # OCR modelini yükle
        reader = easyocr.Reader(['en'])
        imgH , imgW,_ = img.shape#resmin boyutlarını alıyoruz. x,y,height,width bilgileri

        # Metinleri tanı
        result = reader.readtext(img)

        # Algılanan metinleri kare içine al ve orijinal resim üzerine çiz
        for detection in result:
            points = detection[0]  # Algılanan metnin köşe noktalarını al

            min_coordinates = np.min(points, axis=0)
            max_coordinates = np.max(points, axis=0)
            x,y,w,h = int(min_coordinates[0]),int(min_coordinates[1]),int(max_coordinates[0]),int(max_coordinates[1])    

            cv2.rectangle(img,(x, y),(w, h),(0,0,255),5)#her harf için kutu çizdiriyoruz.
            cv2.putText(img,detection[1],(x,y+10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    

        # resmi kaydet
        output_path = 'C:\\Users\\Zerrin Baycan\\Desktop\\testresim\\morphologicalProcesses\\OCR\\' + dosya_Adi
        cv2.imwrite(output_path, img)
    except:
        print("ApplyOcr Hata")








"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from skimage import data
from skimage.color import rgb2hed, hed2rgb

# Example IHC image
ihc_rgb = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\treshold\\GAUSSIAN_150_20.jpg")#resmi okuyoruz#data.immunohistochemistry()

# Separate the stains from the IHC image
ihc_hed = rgb2hed(ihc_rgb)

# Create an RGB image for each of the stains
null = np.zeros_like(ihc_hed[:, :, 0])
ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

# Display
fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(ihc_rgb)
ax[0].set_title("Original image")
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\Original_image.jpg",ihc_rgb)

ax[1].imshow(ihc_h)
ax[1].set_title("Hematoxylin")
cv2.imshow("",ihc_h)
cv2.waitKey(0)
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\Hematoxylin.jpg",ihc_h)

ax[2].imshow(ihc_e)
ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\Eosin.jpg",ihc_e)

ax[3].imshow(ihc_d)
ax[3].set_title("DAB")
cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\DAB.jpg",ihc_d)

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()


######################################################################
# Now we can easily manipulate the hematoxylin and DAB channels:

from skimage.exposure import rescale_intensity

# Rescale hematoxylin and DAB channels and give them a fluorescence look
h = rescale_intensity(
    ihc_hed[:, :, 0],
    out_range=(0, 1),
    in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)),
)
d = rescale_intensity(
    ihc_hed[:, :, 2],
    out_range=(0, 1),
    in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)),
)

# Cast the two channels into an RGB image, as the blue and green channels
# respectively
zdh = np.dstack((null, d, h))

fig = plt.figure()
axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
axis.imshow(zdh)

cv2.imwrite("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\Stain-separated_image_rescaled.jpg",zdh)

axis.set_title('Stain-separated image (rescaled)')
axis.axis('off')
plt.show()
"""

"""
#resim üzerindeki karakterleri kare içine aldırmayı sağlar.
import pytesseract
import cv2
import easyocr

#pyteseract yolunu her zaman yazıyoruz
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"

def ApplyOcr(img):
    try:
        #karakterleri tespit ediyoruz
        imgH , imgW,_ = img.shape#resmin boyutlarını alıyoruz. x,y,height,width bilgileri
        config = r'--oem 3 --psm 6 tessedit_write_images true -l eng+fra'# Örneğin, İngilizce (eng) ve Fransızca (fra) dil modelleri
        boxes = pytesseract.image_to_boxes(img)#her karakterin koordinatlarıyla, genişlik ve yüksekliğini belirliyoruz
        pytesseract.pytesseract.run_and_get_output(img,"jpg",any,config)

        pytesseract.pytesseract.ima

        for b in boxes.splitlines(): 
            
            #Array liste dönüştürüyoruz
            #x 218 51 224 58 0    =>     ['x', '218', '51', '224', '58', '0']
                   
            b= b.split(' ')
            print(b)
            #array listten her bir karaktere kutu çizmek için koordinatları alıyoruz
            x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
            print(x,y,w,h)

            cv2.rectangle(img,(x,imgH - y),(w,imgH - h),(0,0,255),1)#her harf için kutu çizdiriyoruz.
            cv2.putText(img,b[0],(x,imgH - y+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            
        cv2.imshow("resim",img)
        cv2.waitKey(0)
    except:
        print("ApplyOcr Hata")

img = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\treshold\\GAUSSIAN_150_20.jpg")#resmi okuyoruz

h, w = img.shape[:2]
img = cv2.resize(img,(w//4,h//4))

cv2.imshow("resim1",img)
cv2.waitKey(0)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("resim2",img)
cv2.waitKey(0)

ApplyOcr(img)

"""

""" 
##################easyocr##################
import easyocr
import cv2
import numpy as np

# OCR modelini yükle
reader = easyocr.Reader(['en'])

# Resmi oku
image_path = 'C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\treshold\\GAUSSIAN_150_18.jpg'
image = cv2.imread(image_path)
imgH , imgW,_ = image.shape#resmin boyutlarını alıyoruz. x,y,height,width bilgileri

# Metinleri tanı
result = reader.readtext(image_path)

# Algılanan metinleri kare içine al ve orijinal resim üzerine çiz
for detection in result:
    points = detection[0]  # Algılanan metnin köşe noktalarını al

    min_coordinates = np.min(points, axis=0)
    max_coordinates = np.max(points, axis=0)
    x,y,w,h = int(min_coordinates[0]),int(min_coordinates[1]),int(max_coordinates[0]),int(max_coordinates[1])    

    cv2.rectangle(image,(x, y),(w, h),(0,0,255),5)#her harf için kutu çizdiriyoruz.
    cv2.putText(image,detection[1],(x,y+10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    

# resmi kaydet
output_path = 'C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\11111.jpg'
cv2.imwrite(output_path, image)
"""