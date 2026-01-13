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
