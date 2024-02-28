pip install -qr requirements.txt  # install dependencies (ignore errors)

#eğitim için komut satırı
python train.py --img 416 --batch 16 --epochs 500 --data data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache 


#detection için komut satırı
python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source data/images/video2.mp4   


# Her aşamanın test edilmesi için kodun altında command halinde kod satırları vardır bu satırları açarak test edebilirsiniz.

*************************************** Proje İçindeki Dosyalar ********************************

detect.py 

    Bu dosyada lastiğin olduğu kısmı kırpıp <<<<<<< data\\images\\detectimages\crop >>>>>>> dosyası oluşturup bu dosyaya kaydediyoruz. 
    Her detection için ayrı bir crop dosyası oluşturdum. Her işlem adımını crop klasörü  içerisinde klasörlere kaydediyorum 

applyTireFlat.py

    Bu dosyada "crop" klasörü içerisindeki lastik resimleri okunur. 
    Bu resimler düz hale getirilerek <<<<<<< data\\images\\detectimages\crop\flattire >>>>>>> klasörü içine kaydedilir.
    Düzleştirilmiş resimlere morfolojik işlemler uygulanması için applyMorphologicalProcessmetodu çağrılır.

applyMorphologicalProcess.py

    ******************** Geliştirme Devam Ediyor *****************************
    applyTireFlat.py dosyası içinden çağrılır.Lastik düz hale getirildikten sonra yazıları belirgin hale getirmek için morfolojik işlemler uygulanır.
    Yapılan morfolojik işlemler "data\images\detectimages\crop\morphologicalProcesses" altında klasörlenir.
    Lastiğin üzerindeki yazıları ön plana çıkarmak için ilk olarak histogram eşitleme yaparak kontrastı arttırdım. "data\images\detectimages\crop\morphologicalProcesses\Histequalize" klasörüne kaydedilir.
    Histogram eşitleme yapılan resimler üzerinde Clahe histogram(Contrast Limited Adaptive Histogram Equalization) uyguladım. 
    Bu şekilde yazı olan alanlar lastik yüzeyinde daha belirgin ve ayırt edilebilir hale geldi."data\images\detectimages\crop\morphologicalProcesses\ClacheHistogram" klasörüne kaydedilir.
    

applyOCR.py

    applyMorphologicalProcess.py dosyası içinden çağrılır. 
    easyOCR kullanılacaktır. Yazılar belirgin hale getirildikten sonra bu dosya içindeki metod ile lastik üzerindeki metinler tespit edilecektir.


morphologyexercises.py

    Burda lastik üzerinde denenen tüm işlemler test edilebilir.Test için yazılan kodlar da mevcuttur.


Kendime not:

***Resim yoksa uyarı vermek için:
    assert img is not None, "file could not be read, check with os.path.exists()"
