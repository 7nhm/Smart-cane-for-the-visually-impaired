## 9. Phân tích chi tiết 100% về FaceNet

### a. Tổng quan FaceNet
- FaceNet là một kiến trúc deep learning do Google phát triển (2015) để ánh xạ ảnh khuôn mặt thành vector đặc trưng (embedding) trong không gian Euclidean.
- Mục tiêu: các ảnh cùng một người sẽ có embedding gần nhau, khác người sẽ xa nhau.
- Được sử dụng rộng rãi cho nhận diện, xác thực, phân nhóm khuôn mặt.

### b. Kiến trúc mạng
- Backbone phổ biến: Inception-ResNet, Inception-v3, hoặc ResNet-34/50.
- Đầu ra: vector 128 hoặc 512 chiều (tuỳ cấu hình, Facenet512 là 512).
- Không dùng softmax ở cuối, mà dùng trực tiếp vector embedding.

### c. Loss function: Triplet Loss
- Triplet Loss là chìa khoá của FaceNet.
- Mỗi lần huấn luyện, lấy 3 ảnh:
   - Anchor (A): ảnh gốc
   - Positive (P): cùng người với Anchor
   - Negative (N): khác người
- Hàm loss: L = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha, 0)
   - f(x) là embedding của ảnh x
   - alpha là margin (thường 0.2)
- Ý nghĩa: ép embedding của cùng người gần nhau, khác người cách nhau ít nhất alpha.

### d. Pipeline sử dụng trong hệ thống
1. Ảnh khuôn mặt (đã crop, align) được resize về kích thước chuẩn (thường 160x160 hoặc 112x112).
2. Ảnh được chuẩn hoá (normalize pixel value).
3. Truyền qua backbone (Inception-ResNet/ResNet) để lấy vector embedding 512 chiều.
4. Embedding này được lưu vào database, hoặc dùng để so sánh với các embedding khác bằng cosine similarity.

### e. So sánh embedding
- Dùng cosine similarity hoặc Euclidean distance để đo độ giống nhau giữa hai embedding.
- Trong hệ thống này, sử dụng cosine similarity:
   - similarity = (a . b) / (||a|| * ||b||)
- Nếu similarity vượt ngưỡng, xác định là cùng người.

### f. Ưu điểm của FaceNet
- Không cần training classifier cho từng người, chỉ cần lưu embedding.
- Có thể thêm người mới mà không cần huấn luyện lại toàn bộ mạng.
- Độ chính xác cao, robust với nhiều điều kiện ánh sáng, góc mặt.
- Có thể dùng cho clustering, xác thực, tìm kiếm khuôn mặt.

### g. Nhược điểm và lưu ý
- Cần ảnh crop và align khuôn mặt chuẩn xác, nếu không embedding sẽ kém chất lượng.
- Nếu ảnh đầu vào bị mờ, nghiêng quá nhiều, che khuất, embedding có thể không ổn định.
- Cần RAM để lưu cache embedding nếu số lượng lớn.
- Nếu dùng backbone lớn (Inception-ResNet), inference sẽ chậm hơn các model nhẹ.

### h. Ứng dụng thực tế trong hệ thống
- Sử dụng model Facenet512 (DeepFace) để trích xuất embedding cho mọi ảnh khuôn mặt.
- Embedding được lưu vào SQLite, đồng bộ RAM cache để tăng tốc so sánh.
- Khi nhận diện, chỉ cần so sánh embedding mới với embedding đã lưu, không cần huấn luyện lại.
- Có thể mở rộng cho xác thực, phân nhóm, tìm kiếm khuôn mặt tương tự.

### i. Tài liệu tham khảo
- [FaceNet: A Unified Embedding for Face Recognition and Clustering (Google, 2015)](https://arxiv.org/abs/1503.03832)
- [DeepFace (thư viện Python)](https://github.com/serengil/deepface)
# VISION-MATE: Hệ Thống Nhận Diện Khuôn Mặt & Quản Lý Thiết Bị

## Tổng Quan
VISION-MATE là hệ thống nhận diện khuôn mặt thời gian thực, kết hợp quản lý thiết bị, cảnh báo, lịch sử nhận diện và quản lý người dùng. Hệ thống sử dụng DeepFace (Facenet512), OpenCV, SQLite và Flask để cung cấp giải pháp nhận diện khuôn mặt, phát hiện vật cản, cảnh báo và lưu trữ lịch sử.

---

## Thành Phần Chính

### 1. Nhận Diện Khuôn Mặt
- **Model/face_model.py**: Xử lý nhận diện khuôn mặt, trích xuất embedding, so sánh với database, lưu lịch sử. Sử dụng nhiều thuật toán phát hiện khuôn mặt (Haar, MTCNN, SSD, RetinaFace) thông qua DeepFace để tăng độ chính xác.
- **Model/face_db.py**: Quản lý database khuôn mặt (thêm, xóa, tìm kiếm, đồng bộ cache).
- **Model/history_db.py**: Lưu lịch sử nhận diện (ai, khi nào, thiết bị nào, ảnh nào, độ tin cậy).
- **uploads/face_crops/**: Lưu ảnh crop khuôn mặt đã nhận diện.
- **rtsp_stream_dual.py, rtsp_stream.py, rtsp_bk.py**: Luồng nhận diện từ camera RTSP, phát hiện khuôn mặt, nhận diện, gửi thông báo.

### 2. Quản Lý Người Dùng
- **Model/user.py**: Đăng ký, đăng nhập, xác thực, cập nhật, xóa người dùng.

### 3. Quản Lý Thiết Bị & Vị Trí
- **Model/device_db.py**: Quản lý thiết bị (thêm, xóa, cập nhật trạng thái, vị trí, heartbeat).
- **Model/location_model.py**: Lưu và truy xuất vị trí thiết bị.

### 4. Cảnh Báo & Lịch Sử
- **Model/alert_db.py**: Lưu và truy xuất cảnh báo.
- **Model/history_db.py**: Lưu lịch sử nhận diện khuôn mặt.

### 5. Phát Hiện Vật Cản
- **Model/advanced_obstacle_detector.py**: Phát hiện vật cản bằng nhiều phương pháp (background subtraction, edge, motion, cascade).

---

## Quy Trình Nhận Diện Khuôn Mặt
1. **Phát hiện khuôn mặt** trên frame từ camera:
   - Trên luồng realtime: dùng OpenCV Haar Cascade để phát hiện nhanh.
   - Khi trích xuất embedding hoặc crop khuôn mặt (DeepFace): thử nhiều backend detector (Haar, MTCNN, SSD, RetinaFace) để chọn ra khuôn mặt tốt nhất, tăng độ chính xác nhận diện.
2. **Cắt và lưu ảnh khuôn mặt** vào uploads/face_crops.
3. **Trích xuất embedding** bằng DeepFace (Facenet512).
4. **So sánh embedding** với database (cosine similarity).
5. **Xác định danh tính** (nếu vượt ngưỡng, trả về tên; ngược lại UNKNOWN).
6. **Lưu lịch sử** vào database (ai, thiết bị, độ tin cậy, ảnh, thời gian).
7. **Gửi thông báo** (nếu cần) tới thiết bị hoặc server.

---

## Các Đối Tượng Chính & Vai Trò
- **Face**: Thông tin khuôn mặt (id, tên, mối quan hệ, email, ảnh, embedding).
- **User**: Người dùng hệ thống (id, tên, email, mật khẩu, v.v.).
- **Device**: Thiết bị camera hoặc IoT (id, tên, vị trí, trạng thái, heartbeat).
- **History**: Lịch sử nhận diện (ai, khi nào, thiết bị nào, ảnh nào, độ tin cậy).
- **Alert**: Cảnh báo (tiêu đề, nội dung, thời gian).
- **Location**: Vị trí thiết bị (lat, lng, địa chỉ, thời gian).

---

## Câu Hỏi & Tự Trả Lời

### 1. Làm sao hệ thống nhận diện khuôn mặt?
- Sử dụng DeepFace (Facenet512) để trích xuất embedding từ ảnh crop khuôn mặt, so sánh với embedding đã lưu trong database bằng cosine similarity. Nếu similarity vượt ngưỡng, xác định là người quen, ngược lại là UNKNOWN.

### 2. Dữ liệu khuôn mặt được lưu ở đâu?
- Ảnh crop khuôn mặt lưu ở uploads/face_crops/. Thông tin embedding, tên, ảnh, mối quan hệ... lưu trong bảng faces của SQLite (visionmate.db).

### 3. Làm sao thêm/xóa khuôn mặt?
- Thêm: Gọi hàm insert_face trong Model/face_db.py với embedding, tên, ảnh, v.v. Xóa: Gọi delete_face với id, đồng thời xóa file ảnh và đồng bộ lại cache.

### 4. Lịch sử nhận diện được lưu thế nào?
- Mỗi lần nhận diện (dù thành công hay UNKNOWN), hệ thống lưu bản ghi vào bảng face_history (ai, thiết bị, độ tin cậy, ảnh, thời gian, có phải ảnh crop không).

### 5. Làm sao đồng bộ cache khuôn mặt?
- Hàm sync_faces_cache trong Model/face_model.py sẽ tải lại toàn bộ embedding từ database vào RAM để tăng tốc nhận diện.

### 6. Có thể nhận diện nhiều người cùng lúc không?
- Hệ thống hiện tại nhận diện từng khuôn mặt lớn nhất trên frame. Có thể mở rộng để nhận diện nhiều khuôn mặt nếu cần.

### 7. Làm sao phát hiện vật cản?
- Sử dụng Model/advanced_obstacle_detector.py với nhiều phương pháp: background subtraction, edge, motion, cascade. Kết quả trả về bounding box, độ tin cậy, mức nguy hiểm.

### 8. Làm sao quản lý thiết bị?
- Thêm/xóa/cập nhật thiết bị qua Model/device_db.py. Theo dõi trạng thái online/offline qua heartbeat.

### 9. Làm sao quản lý người dùng?
- Đăng ký, đăng nhập, xác thực, cập nhật, xóa qua Model/user.py. Mật khẩu được hash an toàn.

### 10. Làm sao lấy lịch sử nhận diện hoặc cảnh báo?
- Dùng Model/history_db.py (lấy lịch sử nhận diện) và Model/alert_db.py (lấy cảnh báo mới nhất).

---

## Cài Đặt & Chạy
1. Cài Python >=3.8, pip.
2. Cài các thư viện:
   ```
   pip install -r requirements.txt
   ```
3. Chạy server Flask hoặc script nhận diện:
   ```
   python app.py
   # hoặc
   python rtsp_stream.py
   ```

---

## Kiến Trúc Tổng Quan
- **Flask**: Web server, API, giao diện quản lý.
- **OpenCV**: Xử lý ảnh, phát hiện khuôn mặt, vật cản.
- **DeepFace**: Trích xuất embedding, nhận diện khuôn mặt.
- **SQLite**: Lưu trữ thông tin khuôn mặt, người dùng, thiết bị, lịch sử.
- **SocketIO**: Giao tiếp thời gian thực (nếu dùng dashboard).

---

## Liên Hệ & Đóng Góp
- Đóng góp, báo lỗi: Tạo issue hoặc pull request.
- Liên hệ: [your-email@example.com]

---

## Ghi chú
- Đảm bảo camera RTSP hoạt động và cấu hình đúng URL trong các file rtsp_*.py.
- Đảm bảo các thư mục uploads/, uploads/face_crops/, uploads/rtsp_faces/ tồn tại và có quyền ghi.
- Để nhận diện tốt, nên thêm nhiều mẫu khuôn mặt với các góc khác nhau.
