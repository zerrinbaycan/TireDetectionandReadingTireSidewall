#import pymssql
import pyodbc


# Bağlantı bilgilerini ayarlayın
server = '10.10.92.2'
database = 'TireDetect'
username = 'Usr_WiseAdmin'
password = 'Wise_3896!'
 # Bağlantı dizesini tanımlıyoruz
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'



def FindTireInfoFromOCRText(ocrlist):
   
   # Veritabanına bağlandık
    conn = pyodbc.connect(conn_str)

    # Bağlantı üzerinden bir cursor oluşturuyoruz
    cursor = conn.cursor()

    sql = """
            select * from FindTireInfoFromOCRText (?)
          """  
    cursor.execute(sql,ocrlist)
    
    # Sonuçları alıyoruz
    results = cursor.fetchall()


    # Cursor ve bağlantıyı kapatıyoruz
    cursor.close()
    conn.close()

    return results























"""
def getallBrands():
    select * from TireBrand where IsDeleted = 0 and IsCappedCasing = 0 order by Name


def getallSizes():

select T.Size
from Tire T(nolock)
	 INNER JOIN TireBrand TB(NOLOCK) ON TB.Id = T.TireBrandId
where TB.IsDeleted = 0 and TB.IsCappedCasing = 0 AND T.IsDeleted = 0 AND RTRIM(LTRIM(Size)) <> ''
GROUP BY T.Size
ORDER BY T.Size

def getallPatterns():
select T.Pattern
from Tire T(nolock)
	 INNER JOIN TireBrand TB(NOLOCK) ON TB.Id = T.TireBrandId
where TB.IsDeleted = 0 and TB.IsCappedCasing = 0 AND T.IsDeleted = 0 AND RTRIM(LTRIM(T.Pattern)) <> ''
GROUP BY T.Pattern
ORDER BY T.Pattern
"""    





#Bu metod bütün lastik listesini getirmeyi sağlar
def getTireInfo():
   
   # Veritabanına bağlandık
    conn = pyodbc.connect(conn_str)

    # Bağlantı üzerinden bir cursor oluşturuyoruz
    cursor = conn.cursor()

    #Tüm lastiklerin listesini çekiyoruz  
    cursor.execute("select T.Id,Marka = TB.Name,Model = T.Pattern, Ebat = T.Size from Tire T(NOLOCK) \
                    INNER JOIN TireBrand  TB (NOLOCK) ON TB.Id = T.TireBrandId \
                    where T.IsDeleted = 0 \
                    order by 1 ,2 ,3 ")
    # Sonuçları alıyoruz
    results = cursor.fetchall()

    results_list = []
    # Sonuçları işleme
    for row in results:
        row_dict = {'Id': row.Id, 'Brand': row.Marka, 'Pattern': row.Model, 'Size': row.Ebat}
        results_list.append(row_dict)

    # Cursor ve bağlantıyı kapatıyoruz
    cursor.close()
    conn.close()

    return results_list


#Bu metod ile ocr ile bulunan texti içeren marka, model, size var mı kontrol ediyoruz
def getBrandPatternSizefromText(text):

    # Veritabanına bağlanıyoruz
    conn = pyodbc.connect(conn_str)

    # Bağlantı üzerinden bir cursor oluşturuyoruz
    cursor = conn.cursor()
    
    # Örnek bir SQL sorgusu çalıştırın
    cursor.execute("select T.Id,Marka = TB.Name,Model = T.Pattern, Ebat = T.Size from Tire T(NOLOCK) \
                    INNER JOIN TireBrand  TB (NOLOCK) ON TB.Id = T.TireBrandId \
                    where T.IsDeleted = 0 and (TB.Name COLLATE Latin1_General_CI_AI LIKE ? \
                    or T.Pattern COLLATE Latin1_General_CI_AI LIKE ? \
                    or T.Size LIKE ? )\
                    order by 1 ,2 ,3 ",('%' + text + '%'),('%' + text + '%'),('%' + text + '%'))
    
    # Sonuçları alın
    results = cursor.fetchall()

    # Cursor ve bağlantıyı kapatın
    cursor.close()
    conn.close()

    return results




#Bu metod Marka tablosunda gönderilen parametreyi içeren markaları döner
def getBrand(marka,array,resultarray):

    # Veritabanına bağlanıyoruz
    conn = pyodbc.connect(conn_str)

    # Bağlantı üzerinden bir cursor oluşturuyoruz
    cursor = conn.cursor()
    
    # Örnek bir SQL sorgusu çalıştırın
    cursor.execute("select T.Id,Marka = TB.Name,Model = T.Pattern, Ebat = T.Size from Tire T(NOLOCK) \
                    INNER JOIN TireBrand  TB (NOLOCK) ON TB.Id = T.TireBrandId \
                    where T.IsDeleted = 0 and TB.Name COLLATE Latin1_General_CI_AI LIKE ? \
                    order by 1 ,2 ,3 ",('%' + marka + '%',))
    
    # Sonuçları alın
    results = cursor.fetchall()

    # Sonuçları işleyin
    for row in results:
        array.append(row)
        resultarray.append(row)

    # Cursor ve bağlantıyı kapatın
    cursor.close()
    conn.close()

    return results


#Bu metod lastik tanım tablosunda gönderilen parametreyi içeren patternleri döner
def getPattern(model,array,resultarray):
    # Bağlantı dizesini oluşturun
    conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

    # Veritabanına bağlan
    conn = pyodbc.connect(conn_str)

    # Bağlantı üzerinden bir cursor oluşturun
    cursor = conn.cursor()

    # Örnek bir SQL sorgusu çalıştırın
    cursor.execute("select T.Id,Marka = TB.Name,Model = T.Pattern, Ebat = T.Size from Tire T(NOLOCK) \
                    INNER JOIN TireBrand  TB (NOLOCK) ON TB.Id = T.TireBrandId \
                    where T.IsDeleted = 0 and T.Pattern COLLATE Latin1_General_CI_AI LIKE ? \
                    order by 1 ,2 ,3 ",('%' + model + '%',))
    # Sonuçları alın
    results = cursor.fetchall()

    # Sonuçları işleyin
    for row in results:
        array.append(row)
        resultarray.append(row)

    # Cursor ve bağlantıyı kapatın
    cursor.close()
    conn.close()

    return results

#Bu metod lastik tanım tablosunda gönderilen parametreyi içeren size bilgisini döner
def getSize(ebat,array,resultarray):
    # Bağlantı dizesini oluşturun
    conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

    # Veritabanına bağlan
    conn = pyodbc.connect(conn_str)

    # Bağlantı üzerinden bir cursor oluşturun
    cursor = conn.cursor()

    # Örnek bir SQL sorgusu çalıştırın
    cursor.execute("select T.Id,Marka = TB.Name,Model = T.Pattern, Ebat = T.Size from Tire T(NOLOCK) \
                    INNER JOIN TireBrand  TB (NOLOCK) ON TB.Id = T.TireBrandId \
                    where T.IsDeleted = 0  and T.Size LIKE ? \
                    order by 1 ,2 ,3 ",('%' + ebat + '%',))
    # Sonuçları alın
    results = cursor.fetchall()

    # Sonuçları işleyin
    for row in results:
        array.append(row)
        resultarray.append(row)

    # Cursor ve bağlantıyı kapatın
    cursor.close()
    conn.close()

    return results



def findTireInfo(ocrlist):
   
   # Veritabanına bağlandık
    conn = pyodbc.connect(conn_str)

    # Bağlantı üzerinden bir cursor oluşturuyoruz
    cursor = conn.cursor()

    sql = """
            select * from FindTireInfo (?)
          """  
    cursor.execute(sql,ocrlist)
    
    # Sonuçları alıyoruz
    results = cursor.fetchall()


    # Cursor ve bağlantıyı kapatıyoruz
    cursor.close()
    conn.close()

    return results




"""
#Kelime metin içindeki kelimelerde var mı kontrol etmeyi sağlıyor
def kelimeyi_bul(kelime, kelime_listesi):
    bulunan_kelimeler = []
    for kelime in kelime_listesi:
        harf_sayisi = 0
        for harf in kelime:
            if harf in karisik_kelime:
                harf_sayisi += 1
        if harf_sayisi == len(kelime):
            bulunan_kelimeler.append(kelime)
    return bulunan_kelimeler

#Aranan kelimedeki her harf metin içindeki kelimede var mı kontrol etmeyi sağlıyor. Tüm harfler varsa listeye ekliyoruz
def kelimeyi_bul2(kelime, kelime_listesi):
    bulunan_kelimeler = []
    for kelime in kelime_listesi:
        harf_sayisi = 0
        for harf in karisik_kelime:
            if harf in kelime:
                harf_sayisi += 1
        if harf_sayisi == len(karisik_kelime):
            bulunan_kelimeler.append(kelime)
    return bulunan_kelimeler

karisik_kelime = "mkaa"
kelime_listesi = ["kam", "mak", "ama", "aka", "karma"]

bulunan_kelimeler = kelimeyi_bul(karisik_kelime, kelime_listesi)
bulunan_kelimeler2 = kelimeyi_bul2(karisik_kelime, kelime_listesi)
print("Bulunan kelimeler:", bulunan_kelimeler)
"""