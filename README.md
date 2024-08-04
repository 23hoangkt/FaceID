# Tổng quan 
FaceID bao gồm Face Recognition và Liveness Detection 
![Example Image](img.jpg)


FaceID dựa trên FaceNet và MTCNN
Liveness Detection dựa trên OpenCV và DeepLearning 







# Dowload Models nhận dạng khuôn mặt tại 
https://drive.google.com/drive/folders/11awWC9KeSZyqTL9ivn-YmxgGAvPpyuw0?usp=sharing
# Cài đặt môi trường chạy venv 
```pip install -r requirements.txt```
# Chạy 
```python\src```
```python liveness.py --model liveness.model.h5 --le le.pickle --detector face_detector```
```python3 liveness.py --model liveness.model.h5 --le le.pickle --detector face_detector```
# Train lại với bộ dữ liệu mới
Tạo bộ ảnh mới, lưu tên là file thư mục chứ ảnh người đó tại ```FaceID\DataSet\FaceData\raw```

# Preprocesscing với dữ liệu mới 
```python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25```

# Tiến hành train
```python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000```
