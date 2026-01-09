# Giải thích từng dòng code file Model/face_model.py

---

## 1. Import và thiết lập ban đầu
```python
import os  # Xử lý đường dẫn, file
import sqlite3  # Kết nối và thao tác với SQLite
import numpy as np  # Xử lý toán học, vector
import json  # Đọc/ghi dữ liệu JSON
import cv2  # Xử lý ảnh với OpenCV
import time  # Xử lý thời gian
import threading  # Đa luồng
from deepface import DeepFace  # Thư viện nhận diện khuôn mặt
import traceback  # In stack trace khi lỗi
from concurrent.futures import ThreadPoolExecutor, TimeoutError  # Đa luồng nâng cao
import warnings  # Quản lý cảnh báo
import datetime  # Xử lý ngày giờ
import sys  # Thao tác với hệ thống
```

## 2. Thiết lập biến toàn cục và đường dẫn
```python
warnings.filterwarnings('ignore')  # Tắt cảnh báo không cần thiết

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Thư mục hiện tại
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Thư mục gốc dự án
DB_PATH = os.path.join(PROJECT_ROOT, "visionmate.db")  # Đường dẫn DB
UPLOADS_ROOT = os.path.join(PROJECT_ROOT, "uploads")  # Thư mục uploads
FACE_CROPS_DIR = os.path.join(UPLOADS_ROOT, "face_crops")  # Thư mục lưu ảnh crop
```

## 3. In thông tin và khởi tạo thư mục/model
```python
print(f"[FACE_MODEL] Project root: {PROJECT_ROOT}")
print(f"[FACE_MODEL] Face crops dir: {FACE_CROPS_DIR}")
os.makedirs(FACE_CROPS_DIR, exist_ok=True)  # Tạo thư mục nếu chưa có

print("Loading DeepFace model (Facenet512)...")
MODEL_NAME = "Facenet512"  # Chọn backbone
try:
	model = DeepFace.build_model(MODEL_NAME)  # Khởi tạo model
	print("DeepFace model ready!")
except Exception as e:
	print(f"Error loading DeepFace model: {e}")
	model = None
```

## 4. Kết nối với history_db để lưu lịch sử
```python
def init_history_connection():
	...
save_history_func = init_history_connection()  # Hàm lưu lịch sử nhận diện
```
Hàm này cố gắng import các hàm lưu lịch sử từ history_db. Nếu lỗi sẽ tạo hàm fallback chỉ in ra console.

## 5. Khởi tạo cache RAM cho embedding
```python
faces_cache = []  # Danh sách embedding khuôn mặt
cache_lock = threading.Lock()  # Đảm bảo thread-safe
cache_version = 1
last_cache_update = time.time()
cache_sync_interval = 10  # Giây, tự động sync lại cache
```

## 6. Hàm lấy dữ liệu khuôn mặt từ database
```python
def get_faces_from_db():
	...
```
Kết nối DB, lấy embedding, tên, id... từ bảng faces. Chuyển embedding từ JSON sang numpy array.

## 7. Hàm đồng bộ cache
```python
def sync_faces_cache():
	...
def get_cache_info():
	...
def check_and_sync_cache_if_needed():
	...
```
Đồng bộ cache từ DB, lấy thông tin cache, tự động sync nếu cache quá cũ.

## 8. Hàm nhận diện khuôn mặt và lưu lịch sử
```python
def recognize_face_from_frame(frame_path, auto_save=True, min_confidence=0.70):
	...
```
Nhận diện khuôn mặt từ ảnh, cắt crop, trích xuất embedding, so sánh với cache, lưu lịch sử vào DB.

## 9. Hàm lấy embedding từ ảnh bất kỳ
```python
def get_embedding_from_image_file(image_path):
	...
```
Trích xuất embedding từ file ảnh, thử nhiều detector khác nhau.

## 10. Hàm hỗ trợ tính toán
```python
def cosine_similarity(a, b):
	...
```
Tính cosine similarity giữa hai vector embedding.

## 11. Hàm cắt và lưu khuôn mặt
```python
def extract_and_save_face_crop(frame_path, output_dir="uploads/face_crops"):
	...
```
Đọc ảnh, phát hiện khuôn mặt bằng nhiều detector, lấy khuôn mặt tốt nhất, lưu file crop.

## 12. Nhận diện đơn giản cho RTSP
```python
def recognize_face_simple(frame_path):
	...
```
Nhận diện nhanh cho luồng RTSP, tự động lưu lịch sử.

## 13. Các hàm tiện ích khác
```python
def load_faces():
	...
def save_recognition_to_history(name, confidence, image_path=None):
	...
```
Đồng bộ cache, lưu lịch sử nhận diện (fallback).

## 14. Khởi tạo cache khi import module
```python
print("Initializing face cache...")
sync_faces_cache()
```
Khi import file này, cache sẽ được đồng bộ ngay.

## 15. In trạng thái module
```python
print(f"Face model module ready! Cache v{cache_version} with {len(faces_cache)} faces")
print(f"Database saving: ENABLED")
print(f"Face crops directory: {FACE_CROPS_DIR}")
print(f"Face crops exists: {os.path.exists(FACE_CROPS_DIR)}")
```
In trạng thái khởi tạo của module.

---

Nếu bạn muốn giải thích chi tiết từng dòng code trong một hàm cụ thể, hãy chỉ rõ tên hàm hoặc đoạn code!
