#FindTireInfoFromOCRText isminde dbde table return function create ettim. function ile ocr ile bulduğumuz txtleri dbde benzerlik metodu ile sorgulayarak tahmin yapıyoruz.

import _dbConnect as db

#Ocr sonucu bulunan textlerden istediğimiz bilgiler olmayanlar gelirse elemek için
delete_textlist = ["RADIAL","TUBELESS","OUTSIDE","MADE IN","MADE","REGROOVABLE","WARNING"]

def isempty(str):
    if bool(str.strip()):#eğer str dolu ise true döner. bu yüzden true ise yani str dolu ise False dönsün, yani boş değil diyoruz. boş ise True dönsün.
        return False  
    else:
        return True

def OcrTextClear(ocr_text_list):
    filtered_list = []
    for item in ocr_text_list:
        item = item.strip()
        if len(item) >= 3 and item.upper() not in delete_textlist:
            filtered_list.append(item)
    
    return filtered_list

def textsimilaritymain(ocr_text_list):
    ocr_text_list = OcrTextClear(ocr_text_list)
    tempstr = ""
    for text in ocr_text_list:
        text = text.replace("'","''")
        #sqlde XML'e dönüştürürken özel karakterler hata veriyor(&,<,> gibi). Bu yüzden özel karakterlerde düzeltme yapmamız gerekli
        text = text.replace("&","&amp;")
        text = text.replace("<","&lt;")
        text = text.replace(">","&gt;")
        tempstr += "<KEY>" + text + "</KEY>"

    
    ocr_text_list = tempstr

    res = db.FindTireInfoFromOCRText(ocr_text_list)#ocr listesindeki metinlerin marka,model,ebat alanarında eşiti varmı kontrol
    listMarka = [item for item in res if item[1] == "Marka"]
    listModel = [item for item in res if item[1] == "Model"]
    listEbat = [item for item in res if item[1] == "Ebat"]

    print(" Tahmin edilen Marka:{}".format(listMarka))
    print("\n Tahmin edilen Marka:{}".format(listModel))
    print("\n Tahmin edilen Ebat:{}".format(listEbat))


    return res