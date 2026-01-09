# Phân tích chi tiết các thuật toán trong VISION-MATE

## 1. Nhận diện khuôn mặt (Face Recognition)
- **Trích xuất embedding:**
  - Sử dụng DeepFace với model Facenet512 để chuyển ảnh khuôn mặt thành vector đặc trưng (embedding).
  - Thử nhiều backend detector: opencv, retinaface, ssd, mtcnn để tăng khả năng phát hiện khuôn mặt.
- **So sánh embedding:**
  - Sử dụng hàm cosine similarity để đo độ tương đồng giữa embedding của ảnh mới và các embedding đã lưu trong database.
  - Nếu similarity vượt ngưỡng động (thường 0.6–0.7), xác định là người quen, ngược lại là UNKNOWN.
- **Lưu lịch sử:**
  - Mỗi lần nhận diện (dù thành công hay UNKNOWN), lưu bản ghi vào bảng face_history với thông tin: tên, thiết bị, độ tin cậy, ảnh, thời gian, có phải ảnh crop không.

## 2. Phát hiện khuôn mặt (Face Detection)
- **Haar Cascade (OpenCV):**
  - Sử dụng cascade classifier (haarcascade_frontalface_default.xml) để phát hiện khuôn mặt trên frame.
  - Áp dụng các bước tiền xử lý: chuyển grayscale, equalizeHist, GaussianBlur để giảm nhiễu.
  - Lọc false positive bằng cách kiểm tra kích thước, độ ổn định vị trí qua nhiều frame (face tracking bằng deque).

## 3. Phát hiện vật cản (Obstacle Detection)
- **Background Subtraction:**
  - Sử dụng MOG2 để tách foreground/background, tìm contour có diện tích phù hợp, lọc theo aspect ratio và solidity.
- **Edge Density:**
  - Tính gradient (Sobel), lấy magnitude, threshold để tìm vùng có nhiều cạnh, lọc theo mật độ cạnh và diện tích.
- **Motion Detection:**
  - Sử dụng optical flow (Farneback) để phát hiện chuyển động giữa các frame, tìm vùng có motion intensity cao.
- **Cascade Detection:**
  - Dùng các cascade (fullbody, upperbody, lowerbody) để phát hiện vật thể lớn.
- **ROI Filtering:**
  - Chỉ giữ vật cản trong vùng quan tâm (ROI) phía trước camera.
- **Temporal Filtering:**
  - Lọc nhiễu thời gian bằng cách chỉ giữ vật cản xuất hiện liên tiếp nhiều frame.
- **Non-Maximum Suppression:**
  - Loại bỏ các bounding box trùng lặp dựa trên tỉ lệ overlap.
- **Danger Level:**
  - Tính mức độ nguy hiểm dựa trên vị trí, kích thước, độ tin cậy, và việc bounding box chạm biên khung hình.

## 4. Quản lý cache embedding
- **RAM Cache:**
  - Toàn bộ embedding khuôn mặt được load vào RAM để tăng tốc so sánh.
  - Tự động đồng bộ lại cache sau mỗi thao tác thêm/xóa hoặc sau một khoảng thời gian nhất định.

## 5. Các thuật toán phụ trợ
- **Cosine Similarity:**
  - $\text{similarity} = \frac{a \cdot b}{\|a\| \|b\|}$
  - Được dùng để đo độ tương đồng giữa hai vector embedding.
- **Threading:**
  - Sử dụng đa luồng cho camera, nhận diện, gửi thông báo để giảm độ trễ.

## 6. Quản lý thiết bị, người dùng, vị trí
- Chủ yếu sử dụng các thao tác CRUD với SQLite, không có thuật toán phức tạp.

---

## Tổng kết
Các thuật toán chính tập trung vào:
- Phát hiện và nhận diện khuôn mặt nhanh, chính xác, chống nhiễu.
- Phát hiện vật cản đa phương pháp, lọc nhiễu không gian và thời gian.
- Tối ưu tốc độ qua cache, đa luồng, và các ngưỡng động.

Nếu cần phân tích sâu hơn từng hàm hoặc thuật toán cụ thể, hãy chỉ rõ module hoặc chức năng bạn muốn tìm hiểu.