import tensorflow as tf
import facenet
import align.detect_face
import numpy as np
import cv2
import os
import pickle
from sklearn.svm import SVC

def add_new_face(name, image_path, facenet_model_path, classifier_path):
    # Các thông số cần thiết
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    # Tạo đồ thị TensorFlow mới
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # Load mô hình FaceNet
            facenet.load_model(facenet_model_path)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Khởi tạo MTCNN để phát hiện khuôn mặt
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            # Đọc ảnh của người mới
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot read image at path: {image_path}")
                return

            # Resize ảnh nếu cần thiết
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

            # Phát hiện khuôn mặt trong ảnh
            bounding_boxes, _ = align.detect_face.detect_face(image, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            # Nếu không tìm thấy khuôn mặt
            if len(bounding_boxes) == 0:
                print("No face detected in the image.")
                return

            # Trích xuất bounding box của khuôn mặt đầu tiên
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - 22, 0)
            bb[1] = np.maximum(det[1] - 22, 0)
            bb[2] = np.minimum(det[2] + 22, image.shape[1])
            bb[3] = np.minimum(det[3] + 22, image.shape[0])

            # Cắt ảnh khuôn mặt và resize về kích thước INPUT_IMAGE_SIZE
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            scaled = facenet.prewhiten(scaled)
            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

            # Trích xuất embedding từ ảnh khuôn mặt sử dụng FaceNet
            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)

            # Load lại mô hình SVM từ tệp đã lưu
            if os.path.exists(classifier_path):
                with open(classifier_path, 'rb') as file:
                    try:
                        model, class_names, embeddings_list, names_list = pickle.load(file)
                    except ValueError:
                        print("Error loading model. Initializing new model.")
                        model = SVC(kernel='linear', probability=True)
                        class_names = []
                        embeddings_list = []
                        names_list = []
            else:
                model = SVC(kernel='linear', probability=True)
                class_names = []
                embeddings_list = []
                names_list = []

            # Thêm embedding và tên vào danh sách
            embeddings_list.append(emb_array)
            names_list.append(name)
            class_names = list(set(names_list))

            # Chuyển đổi danh sách embedding và tên thành numpy array
            X = np.array(embeddings_list).reshape(-1, emb_array.shape[1])
            y = np.array(names_list)

            # Huấn luyện lại mô hình SVM
            model.fit(X, y)

            # Lưu lại mô hình SVM và các biến liên quan vào tệp
            with open(classifier_path, 'wb') as file:
                pickle.dump((model, class_names, embeddings_list, names_list), file)
            print(f"Successfully added {name} to the system.")

# Hàm chính để chạy chương trình
def main():
    facenet_model_path = 'C:/Users/hoang/Desktop/FaceID/models/20180402-114759.pb'
    classifier_path = 'C:/Users/hoang/Desktop/FaceID/models/facemodel.pkl'
    image_path = 'C:/Users/hoang/Desktop/FaceID/src/lva.jpg'
    add_new_face('NewMember', image_path, facenet_model_path, classifier_path)

if __name__ == "__main__":
    main()
