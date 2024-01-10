pip install -qr requirements.txt  # install dependencies (ignore errors)

python train.py --img 416 --batch 16 --epochs 500 --data data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache #eğitim için komut satırı


python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source data/images/video2.mp4   #detection için komut satırı
