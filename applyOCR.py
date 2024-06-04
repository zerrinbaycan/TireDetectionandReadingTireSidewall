#pytesseract ile yazıları tespit etmeye çalıştım ama başarılı olmadı. easyocr metinleri bulmada çok daha başarılı o yüzden easyocr ile devam edicem
import os
import easyocr
import cv2
import numpy as np
import pytesseract
import _dbConnect as db
import Levenshtein
import findTextsimilaritydb as fts

def yazialaninnibul(img,file_dir,dosya_adi):
    try:
        
        text_list = []#bulduğu metinleri döndüreceğim liste

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # OCR modelini yükle
        reader = easyocr.Reader(['en'])
        imgH , imgW,_ = img.shape#resmin boyutlarını alıyoruz. x,y,height,width bilgileri

        # Metinleri tanı
        result = reader.readtext(img)

        # Algılanan metinleri kare içine al ve orijinal resim üzerine çiz
        i = 0
        for detection in result:
            points = detection[0]  # Algılanan metnin köşe noktalarını al

            min_coordinates = np.min(points, axis=0)
            max_coordinates = np.max(points, axis=0)
            x,y,w,h = int(min_coordinates[0]),int(min_coordinates[1]),int(max_coordinates[0]),int(max_coordinates[1]) 

            i += 1
            yaziolan_alan = img[y-10:h+10,x-10:w+10]
            isim = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop9\\flattire\\yaziliolan_alan\\yaziolan_alan_{}_{}".format(i,dosya_adi)
            cv2.imwrite(isim, yaziolan_alan)

        # resmi kaydet
        file_dir = os.path.join(file_dir, 'OCR')        
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        file_dir = os.path.join(file_dir, dosya_adi)
        cv2.imwrite(file_dir, img)

        return text_list
    except:
        print("ApplyOcr Hata")

def ApplyOcr(img,file_dir,dosya_adi):
    try:
        
        text_list = []#bulduğu metinleri döndüreceğim liste

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
            text_list.append(detection[1])

        # resmi kaydet
        file_dir = os.path.join(file_dir, 'OCR')        
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        file_dir = os.path.join(file_dir, dosya_adi)
        cv2.imwrite(file_dir, img)

        return text_list
    except:
        print("ApplyOcr Hata")

def ApplyOcrteseract(img,file_dir,dosya_adi):
    try:
        #pyteseract yolunu her zaman yazıyoruz
        pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #karakterleri tespit ediyoruz
        imgH , imgW,_ = img.shape#resmin boyutlarını alıyoruz. x,y,height,width bilgileri
        config = r'--oem 3 --psm 6 tessedit_write_images true -l eng+fra'# Örneğin, İngilizce (eng) ve Fransızca (fra) dil modelleri
        boxes = pytesseract.image_to_boxes(img)#her karakterin koordinatlarıyla, genişlik ve yüksekliğini belirliyoruz
        
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
            
        file_dir = os.path.join(file_dir, "teseract_" + dosya_adi)
        cv2.imwrite(file_dir, img)
    except:
        print("ApplyOcr Hata")

#Ocr sonucu bulunan textlerden istediğimiz bilgiler olmayanlar gelirse elemek için
delete_textlist = ["RADIAL","TUBELESS","OUTSIDE","MADE IN"," "]
def OcrTextClear(ocr_text_list):
    for item in ocr_text_list:
        if item.upper() in delete_textlist:
            ocr_text_list.remove(item)
#Ocr sonucu bulunan textlerden istediğimiz bilgiler olmayanlar gelirse elemek için
Brand = ""
Pattern = ""
Size = ""



"""
###############   YÖNTEM 1 ####################################
#BEGIN
#isEbatmi() FONKSİYONU İLE EBATIN TESPİT EDİLMESİ İŞLEMİ

yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\12.jpg" #isEbatmi fonksiyonu için 195/55 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\36.jpg" #isEbatmi fonksiyonu için 385/65 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\105.jpg" #isEbatmi fonksiyonu için 195[65R15 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\52.jpg" #isEbatmi fonksiyonu için 385/55R22.5 ebatı aşgılamak için 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop7\\morphologicalProcesses\\Histequalize2\\IMG_1701.jpg" #isEbatmi fonksiyonu için 385/55R22.5 ebatı algılamak için 

img = cv2.imread(yol,0)
ocr_text_list = ApplyOcr(img,"C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\zerrin","ocr.jpg")
OcrTextClear(ocr_text_list) #kontrol etmememiz gereken kelimeler bulmuşsak eğer bunları siliyoruz.
tire_list = db.getTireInfo()#db'deki lastik bilgilerini çekiyoruz.

#okunan bilgi ebat mı kontrol etttiğimiz blok
EbatTahminleriArray = []

def harf_sayisi(string, harf):
    sayac = 0
    for char in string:
        if char == harf:
            sayac += 1
    return sayac

def isEbatmi(ocr_text_list):
    for text in ocr_text_list:
        text = text.upper()
        if (text[0:3].isnumeric()) and ('/' in text) and ('R' in text) and ((text[(text.index("/") +1):(text.index("R"))]).isnumeric()) and text[(text.index("R")+1) : (text.index("R")+2)].isnumeric():# text formatı 385/55R22.5  ise            
            EbatTahminleriArray.append(text)#buraya daha sonra . karakterini okumamasına karşı kontrol ekle
        elif (text[0:3].isnumeric()) and ('/' in text):# text formatı 385/55 ise
            EbatTahminleriArray.append(text)
        elif ('R' in text) and (harf_sayisi(text,"R") == 1) and  (text[0:(text.index("R"))].isnumeric()) and (text[(text.index("R")+1) : (text.index("R")+2)].isnumeric()):#text formatı 13R22.5  ise
            EbatTahminleriArray.append(text)

isEbatmi(ocr_text_list)
if(len(EbatTahminleriArray) == 1):#sizearray tek elemanlı ise ebat olarak tek bilgi bulmuş demektir.Bunu direk Size'a atabiliriz.
    Size = EbatTahminleriArray[0]
    tire_list = [item for item in tire_list if item["Size"] == Size] #burdan sonra tire_list listesinde marka model ile karşılaştırma yapılabilir.Size'a göre filtrelenmiş oldu
#okunan bilgi ebat mı kontrol etttiğimiz blok
#END
###############   YÖNTEM 1 ####################################
"""

"""
###############   YÖNTEM 2 ####################################
#BEGIN
#Ocr ile bulduğum textleri Databaseden marka,model,ebat kolonlarında içeren satır varmı, varsa resultarray'e ekleyerek bir liste oluşturuyorum.
#Tekrarlayan Id'leri ve tekrar sayılarını bulup yazdırdım. Burdan tekrarlayan IDler arasında marka,model,ebat kolonlarında uniq olan kolon varsa tespit edip (groupby ile olabilir) arasından marka,model,ebat kararlaştırmam gerekli

#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\12.jpg" #isEbatmi fonksiyonu için 195/55 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\36.jpg" #isEbatmi fonksiyonu için 385/65 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\105.jpg" #isEbatmi fonksiyonu için 195[65R15 
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\52.jpg" #isEbatmi fonksiyonu için 385/55R22.5 ebatı aşgılamak için 

img = cv2.imread(yol,0)
ocr_text_list = ApplyOcr(img,"C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\zerrin","ocr.jpg")
OcrTextClear(ocr_text_list) #kontrol etmememiz gereken kelimeler bulmuşsak eğer bunları siliyoruz.
tire_list = db.getTireInfo()#db'deki lastik bilgilerini çekiyoruz.

resultarray = []
for kelime in ocr_text_list:
    if len(kelime) < 2:
            continue
    kelime = kelime.replace("@","")
    
    results = db.getBrandPatternSizefromText(kelime)
    for row in results:
        row_dict = {'Id': row.Id, 'Brand': row.Marka, 'Pattern': row.Model, 'Size': row.Ebat}
        resultarray.append(row_dict)

id_counts = {}
for row_dict in resultarray:
    id_val = row_dict['Id']
    id_counts[id_val] = id_counts.get(id_val, 0) + 1

duplicate_ids = []
for id_val, count in id_counts.items():
    if count > 1:
        print(f"Id: {id_val} - Tekrar Sayısı: {count}")
        duplicate_ids.append(id_val) 
expectedResult = []
for item in resultarray:
    if item['Id'] in duplicate_ids and item['Id'] not in expectedResult:
        expectedResult.append(item)
b = {x['Id']:x for x in expectedResult}.values()
print(b)

#END
###############   YÖNTEM 2 ####################################
"""

"""
###############   YÖNTEM 3 ####################################
#BEGIN
#Ocr ile bulduğum textleri Databaseden marka,model,ebat kolonlarında içeren satır varmı, varsa resultarray'e ekleyerek bir liste oluşturuyorum.
#Bu result array içiende dönerek bulduğum ocr text listesindeki markaya,modele,ebata eşit olan bir text varsa bunları atama yaparak marka,model,ebat kararlaştırması yapıyorum

#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\12.jpg" #isEbatmi fonksiyonu için 195/55 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\36.jpg" #isEbatmi fonksiyonu için 385/65 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\105.jpg" #isEbatmi fonksiyonu için 195[65R15 
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\52.jpg" #isEbatmi fonksiyonu için 385/55R22.5 ebatı aşgılamak için 

img = cv2.imread(yol,0)
ocr_text_list = ApplyOcr(img,"C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\zerrin","ocr.jpg")
OcrTextClear(ocr_text_list) #kontrol etmememiz gereken kelimeler bulmuşsak eğer bunları siliyoruz.
tire_list = db.getTireInfo()#db'deki lastik bilgilerini çekiyoruz.

resultarray = []
for kelime in ocr_text_list:
    if len(kelime) < 2:
            continue
    kelime = kelime.replace("@","")
    
    results = db.getBrandPatternSizefromText(kelime)
    for row in results:
        row_dict = {'Id': row.Id, 'Brand': row.Marka, 'Pattern': row.Model, 'Size': row.Ebat}
        resultarray.append(row_dict)

for tire in resultarray:
    for text in ocr_text_list:
        text = text.upper()
        if Brand == "" and tire["Brand"] == text:
            Brand = text
        elif Pattern == "" and tire["Pattern"] == text:
            Pattern = text
        elif Size == "" and tire["Size"] == text:
            Size = text


filtered_list = [item for item in tire_list if item["Brand"] == Brand and item["Pattern"] == Pattern and item["Size"] == Size]
def filterlist(tire_list,Brand,Pattern,Size):
    for item in tire_list:
        if item["Brand"] == Brand and item["Pattern"] == Pattern and item["Size"] == Size:
            return item
    return None

filtered_list = filterlist(tire_list,Brand,Pattern,Size)
detected = False
if(len(filtered_list) > 0):
    detected = True
#END
###############   YÖNTEM 3 ####################################
"""

"""
###############   YÖNTEM 4 ####################################
#BEGIN
# Gelen ocr_text_list listesindeki bulduğu yazıları marka, model , ebat içinde like ile sorgulayarak her biri için bir liste oluşturdum(markaarray,modelarray,ebatarray).
# markaarray filtrelediğimde tek marka ismi varsa markayı bulmuş oluyorum. modelarray,ebatarray içinde aynı kontrolü yapıyorum

#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\12.jpg" #isEbatmi fonksiyonu için 195/55 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\36.jpg" #isEbatmi fonksiyonu için 385/65 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\105.jpg" #isEbatmi fonksiyonu için 195[65R15 
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\52.jpg" #isEbatmi fonksiyonu için 385/55R22.5 ebatı aşgılamak için 

img = cv2.imread(yol,0)
ocr_text_list = ApplyOcr(img,"C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\zerrin","ocr.jpg")
OcrTextClear(ocr_text_list) #kontrol etmememiz gereken kelimeler bulmuşsak eğer bunları siliyoruz.
tire_list = db.getTireInfo()#db'deki lastik bilgilerini çekiyoruz.


 
resultarray = []
markaarray = []
modelarray = []
ebatarray = []
for kelime in ocr_text_list:
    if len(kelime) < 2:
            continue
    kelime = kelime.replace("@","")
    
    db.getBrand(kelime,markaarray,resultarray)
    db.getPattern(kelime,modelarray,resultarray)
    db.getSize(kelime,ebatarray,resultarray)

Marka = ""
Model = ""
Ebat = ""
markalar = set(veri[1] for veri in markaarray)
if(len(markalar) == 1):
    for marka in markalar:
        Marka = marka

ebatlar = set(veri[3] for veri in ebatarray)
if(len(ebatlar) == 1):
    for ebat in ebatlar:
        Ebat = ebat

modeller = set(veri[2] for veri in modelarray)
if(len(modeller) == 1):
    for model in modeller:
        Model = model
#END
###############   YÖNTEM 4 ####################################
"""

"""
###############   YÖNTEM 5 ####################################
#BEGIN
#tire_list - ocr_text_list karşılaştırmasını for ile yaptım. textin marka model yada ebat içinde olması lazım(sql like kod karşılığı)

#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\12.jpg" #isEbatmi fonksiyonu için 195/55 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\36.jpg" #isEbatmi fonksiyonu için 385/65 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\105.jpg" #isEbatmi fonksiyonu için 195[65R15 
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\52.jpg" #isEbatmi fonksiyonu için 385/55R22.5 ebatı aşgılamak için 

img = cv2.imread(yol,0)
ocr_text_list = ApplyOcr(img,"C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\zerrin","ocr.jpg")
OcrTextClear(ocr_text_list) #kontrol etmememiz gereken kelimeler bulmuşsak eğer bunları siliyoruz.
tire_list = db.getTireInfo()#db'deki lastik bilgilerini çekiyoruz.

#tire_list - ocr_text_list karşılaştırmasını for ile yaptım. textin marka model yada ebat içinde olması lazım(sql like kod karşılığı)
markaarray = []
modelarray = []
ebatarray = []
for tire_info in tire_list:
    marka = tire_info['Brand']  
    model = tire_info['Pattern']  
    size = tire_info['Size']  

    for text in ocr_text_list:
        if len(text) < 2:
            continue

        if text.lower() in marka.lower():  # Büyük-küçük harf duyarlılığı olmadan kontrol et
            markaarray.append(tire_info)

        if text.lower() in model.lower():  # Büyük-küçük harf duyarlılığı olmadan kontrol et
            modelarray.append(tire_info)
        
        if text.lower() in size.lower():  # Büyük-küçük harf duyarlılığı olmadan kontrol et
            ebatarray.append(tire_info)
        

# Benzerlik eşiği
esik_degeri = 0.5  # Dizeler arasındaki benzerlik eşiği
marka_liste = []
model_liste = []
size_liste = []
# İki listeyi karşılaştırma
for tire_info in tire_list:
    marka = tire_info['Brand']  
    model = tire_info['Pattern']  
    size = tire_info['Size'] 

    for kelime in ocr_text_list:
        if len(kelime) < 2:
            continue
        
        benzerlik_orani = Levenshtein.ratio(marka.lower(), kelime.lower())        
        if benzerlik_orani >= esik_degeri:
            marka_liste.append((marka,kelime))

        benzerlik_orani = Levenshtein.ratio(model.lower(), kelime.lower())        
        if benzerlik_orani >= esik_degeri:
            model_liste.append((model,kelime))
            
        benzerlik_orani = Levenshtein.ratio(size.lower(), kelime.lower())        
        if benzerlik_orani >= esik_degeri:
            size_liste.append((size,kelime))


#END
###############   YÖNTEM 5 ####################################
"""

"""
###############   YÖNTEM 6 ####################################
#BEGIN
#Levenshtein Distance Algorithms

yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\52.jpg" #isEbatmi fonksiyonu için 385/55R22.5 ebatı aşgılamak için 

img = cv2.imread(yol,0)
ocr_text_list = ApplyOcr(img,"C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\zerrin","ocr.jpg")
OcrTextClear(ocr_text_list) #kontrol etmememiz gereken kelimeler bulmuşsak eğer bunları siliyoruz.
tire_list = db.getTireInfo()#db'deki lastik bilgilerini çekiyoruz.

param_metin = ""
for txt in ocr_text_list:
    if param_metin != "":
        param_metin += "," 
    temp = "''" + txt + "''"
    param_metin += temp

res = db.findTireInfo(param_metin)#ocr listesindeki metinlerin marka,model,ebat alanarında eşiti varmı kontrol

def levinshteinDistance(str1,str2):
    
    if(str1 == str2):
        return 0
    
    len_str1 = len(str1)
    len_str2 = len(str2)

    if(len_str1 <= 0):
        return len_str2
    if(len_str2 <= 0):
        return len_str1
    
    dp = [[0 for _ in range(len_str2 + 1)] for _ in range(len_str1 + 1)]

    for i in range(0,len_str1+1):
        dp[i][0] = i

    for i in range(0,len_str2+1):
        dp[0][i] = i

    for i in range(1,len_str1+1):
        for j in range(1,len_str2+1):
            if(str1[i-1] == str2[j-1]):
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1+ min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) 

    return dp[len_str1][len_str2]

markaarr = []
modelarr = []
ebatarr = []

for row in tire_list:
    for ocr_text in ocr_text_list:
         val = levinshteinDistance(ocr_text,row["Brand"])
         if val < 3:
            row_dict = {'Id': row.Id, 'Brand': row.Marka, 'Pattern': row.Model, 'Size': row.Ebat}
            results_list.append(row_dict)

#END
###############   YÖNTEM 6 ####################################
"""


###############   YÖNTEM 7 ####################################
#BEGIN
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize2\\105.jpg" 
#yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop10\\flattire\\IMG_2034.jpg" 
yol = "C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\morphologicalProcesses\\Histequalize\\12.jpg" 

img = cv2.imread(yol,0)
ocr_text_list = ApplyOcr(img,"C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop6\\zerrin","ocr.jpg")
ret = fts.textsimilaritymain(ocr_text_list) #kontrol etmememiz gereken kelimeler bulmuşsak eğer bunları siliyoruz.

#END
###############   YÖNTEM 7 ####################################










"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from skimage import data
from skimage.color import rgb2hed, hed2rgb

# Example IHC image
ihc_rgb = cv2.imread("C:\\ZerrinGit\\TireDetectionandReadingTireSidewall\\data\\images\\detectimages\\crop\\morphologicalProcesses\\5_ClacheHistogram.jpg")#resmi okuyoruz#data.immunohistochemistry()

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
