import argparse
import collections
import os
import pickle
import time
import cv2
import imutils
import numpy as np
import tensorflow as tf
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.svm import SVC
import facenet
import align.detect_face
import psutil

# Tạo đối tượng ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument("-m",
                "--model",
                type=str,
                required=True,
                help="path to trained model")
ap.add_argument("-l",
                "--le",
                type=str,
                required=True,
                help="path to label encoder")
ap.add_argument("-d",
                "--detector",
                type=str,
                required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c",
                "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# In ra các đối số đã phân tích để xác minh
print("Arguments:", args)

# Đảm bảo rằng đường dẫn được nhập vào dưới dạng dấu gạch chéo ngược để phù hợp với Windows
detector_path = args["detector"]
model_path = args["model"]
le_path = args["le"]

print("[INFO] loading face detector...")
protoPath = os.path.join(detector_path, "deploy.prototxt")
modelPath = os.path.join(detector_path, "res10_300x300_ssd_iter_140000.caffemodel")

# Kiểm tra sự tồn tại của các file
if not os.path.exists(protoPath):
    print(f"Error: '{protoPath}' does not exist.")
if not os.path.exists(modelPath):
    print(f"Error: '{modelPath}' does not exist.")
if not os.path.exists(model_path):
    print(f"Error: '{model_path}' does not exist.")
if not os.path.exists(le_path):
    print(f"Error: '{le_path}' does not exist.")

# Tải mạng nơ-ron từ file Caffe
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("[INFO] Face detector loaded successfully.")

# Tải mô hình liveness detection và label encoder
print("[INFO] loading liveness detector...")
model = load_model(model_path)
print("[INFO] Liveness model loaded successfully.")

le = pickle.loads(open(le_path, "rb").read())
print("[INFO] Label encoder loaded successfully.")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'C:/Users/hoang/Desktop/FaceID/models/facemodel.pkl'
FACENET_MODEL_PATH = 'C:/Users/hoang/Desktop/FaceID/models/20180402-114759.pb'

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    custom_model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

with tf.Graph().as_default():

    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():

        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(FACENET_MODEL_PATH)

        # Get input and output tensors
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, r'C:\Users\hoang\Desktop\FaceID\src\align')

        people_detected = set()  # theo doi so luong nguoi
        person_detected = collections.Counter()

...

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    faces_found = bounding_boxes.shape[0]

    # Giám sát tài nguyên hệ thống
    cpu_usage = psutil.cpu_percent()
    print(f"Sử dụng CPU: {cpu_usage}%")
    memory_info = psutil.virtual_memory()
    memory_info_dict = dict(memory_info._asdict())
    print("Thông tin bộ nhớ RAM:")
    print(memory_info_dict)
    used_memory_percent = memory_info.percent
    print(f"Bộ nhớ RAM đã sử dụng: {used_memory_percent}%")
    available_memory_percent = memory_info.available * 100 / memory_info.total
    print(f"Bộ nhớ RAM khả dụng: {available_memory_percent}%")

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face_reshaped = face.reshape(1, 32, 32, 3)  # Reshape to match model input shape

            preds = model.predict(face_reshaped)[0]
            j = np.argmax(preds)
            label = le.classes_[j]

            if label == "real":
                if faces_found > 1:
                    cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (255, 255, 255), thickness=1, lineType=2)
                elif faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            # Thêm điều kiện kiểm tra ở đây
                            if cropped.size != 0:
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)

                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = custom_model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                                if best_class_probabilities > 0.85:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)),
                                                (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                else:
                                    name = "Unknown"
                            else:
                                print("Error: Cropped image is empty.")
            else:
                # Vẽ chữ "fake" lên khung hình
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = 'fake'
                org = (50, 50)  # Vị trí của chữ
                font_scale = 1  # Kích thước chữ
                color = (0, 0, 255)  # Màu chữ (BGR) - đỏ
                thickness = 1  # Độ dày của chữ

                cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
                print("Fake")


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
