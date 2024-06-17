# Dowload Models nhận dạng khuôn mặt tại 

#Cài đặt môi trường chạy venv 

<br>
pip install -r requirement.txt
```
#Chạy 
python\src
python liveness.py --model liveness.model.h5 --le le.pickle --detector face_detector
```

<br><br>

#Train lại với bộ dữ liệu mới
Tạo bộ ảnh mới, lưu tên là file thư mục chứ ảnh người đó tại FaceID\DataSet\FaceData\raw

#Preprocesscing với dữ liệu mới 
python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

#Tiến hành train
python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000
