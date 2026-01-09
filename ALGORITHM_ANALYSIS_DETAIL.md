---
##9. Phân tích chi tiết 100% về FaceNet

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
- Loss:
  $$
  L = \max(\|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha, 0)
  $$
  - $f(x)$ là embedding của ảnh $x$
  - $\alpha$ là margin (thường 0.2)
- Ý nghĩa: ép embedding của cùng người gần nhau, khác người cách nhau ít nhất $\alpha$.

### d. Pipeline sử dụng trong hệ thống
1. Ảnh khuôn mặt (đã crop, align) được resize về kích thước chuẩn (thường 160x160 hoặc 112x112).
2. Ảnh được chuẩn hoá (normalize pixel value).
3. Truyền qua backbone (Inception-ResNet/ResNet) để lấy vector embedding 512 chiều.
4. Embedding này được lưu vào database, hoặc dùng để so sánh với các embedding khác bằng cosine similarity.

### e. So sánh embedding
- Dùng cosine similarity hoặc Euclidean distance để đo độ giống nhau giữa hai embedding.
- Trong hệ thống này, sử dụng cosine similarity:
  $$
  	ext{similarity} = \frac{a \cdot b}{\|a\| \|b\|}
  $$
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
# Phân tích chi tiết từng thuật toán trong VISION-MATE
---
## 1. Nhận diện khuôn mặt (Face Recognition)

### a. Trích xuất embedding (DeepFace - Facenet512)

- **Quy trình:**
  1. Ảnh khuôn mặt (đã crop) được truyền vào DeepFace với model Facenet512.
  2. DeepFace thử nhiều backend detector (opencv, retinaface, ssd, mtcnn) để phát hiện khuôn mặt chính xác nhất.
  3. Nếu phát hiện được khuôn mặt, DeepFace trả về vector embedding 512 chiều (float32).
  4. Nếu không phát hiện được, trả về None.
- **Mã nguồn liên quan:**
  - `DeepFace.represent(img_path, model_name, detector_backend, enforce_detection, align)`
  - Thời gian timeout cho mỗi detector: 20 giây.
  - Nếu detector này fail, thử detector khác.

### b. So sánh embedding (Cosine Similarity)

- **Công thức:**
  $$
  \text{similarity} = \frac{\sum_{i=1}^n a_i b_i}{\sqrt{\sum_{i=1}^n a_i^2} \cdot \sqrt{\sum_{i=1}^n b_i^2}}
  $$
- **Ý nghĩa:**
  - similarity gần 1: hai vector rất giống nhau (cùng người)
  - similarity < threshold: khác người
- **Ngưỡng động:**
  - <5 khuôn mặt: threshold = 0.70
  - <10 khuôn mặt: threshold = 0.70–0.72
  - > 10 khuôn mặt: threshold = 0.75–0.8 (có thể chỉnh)
    >
- **Quy trình:**
  1. Lấy embedding của ảnh mới.
  2. So sánh với từng embedding trong cache (RAM).
  3. Chọn similarity cao nhất, nếu vượt ngưỡng thì nhận diện thành công.
  4. Nếu không, trả về UNKNOWN.

### c. Lưu lịch sử nhận diện

- **Bảng:** face_history (id, face_name, device, confidence, image, timestamp, is_cropped_face)
- **Quy trình:**
  - Mỗi lần nhận diện (dù thành công hay UNKNOWN), lưu bản ghi vào bảng này.
  - Nếu có ảnh crop, lưu tên file crop, ngược lại lưu ảnh gốc.

---

## 2. Phát hiện khuôn mặt (Face Detection)

### a. Các thuật toán phát hiện khuôn mặt sử dụng trong hệ thống

#### 1. Haar Cascade (OpenCV)

**Đặc điểm:**

- Cổ điển, rất nhanh, phù hợp realtime, nhưng độ chính xác thấp với góc nghiêng, ánh sáng yếu.
- Sử dụng cho phát hiện khuôn mặt trên luồng RTSP để giảm độ trễ.
  **Quy trình:**
  1. Chuyển frame sang grayscale.
  2. Tiền xử lý: equalizeHist (cân bằng sáng), GaussianBlur (giảm nhiễu).
  3. Dùng `CascadeClassifier.detectMultiScale` với các tham số:
     - scaleFactor: 1.12 (tăng dần kích thước cửa sổ tìm kiếm)
     - minNeighbors: 4 (tăng để giảm false positive)
     - minSize: (60, 60) (lọc mặt quá nhỏ)
  4. Lấy khuôn mặt lớn nhất (nếu có nhiều mặt).
  5. Kiểm tra kích thước tối thiểu (w, h >= 70).
  6. Lưu vị trí các bounding box vào deque (8 frame gần nhất).
  7. Nếu vị trí ổn định qua nhiều frame (di chuyển < 50px), tăng confidence_frames.
  8. Chỉ trả về khi confidence_frames >= 2 (giảm false positive).

#### 2. MTCNN (Multi-task Cascaded Convolutional Networks)

**Đặc điểm:**

- Deep learning, phát hiện tốt nhiều góc mặt, ánh sáng phức tạp, có thể phát hiện nhiều khuôn mặt cùng lúc.
- Chậm hơn Haar, nhưng chính xác hơn nhiều, đặc biệt với ảnh crop hoặc ảnh chất lượng cao.
- Tự động căn chỉnh (align) khuôn mặt.
  **Cách dùng trong hệ thống:**
  - Khi trích xuất embedding hoặc crop khuôn mặt (DeepFace), MTCNN là một trong các backend được thử.
  - Nếu MTCNN phát hiện tốt hơn Haar, sẽ lấy kết quả của MTCNN.

#### 3. SSD (Single Shot Multibox Detector)

**Đặc điểm:**

- Deep learning, phát hiện nhanh, chính xác, hỗ trợ nhiều kích thước khuôn mặt.
- Tốt với ảnh có nhiều khuôn mặt nhỏ hoặc ở xa.
  **Cách dùng trong hệ thống:**
  - Được thử như một backend trong DeepFace khi crop hoặc trích xuất embedding.
  - Nếu SSD phát hiện tốt hơn Haar/MTCNN, sẽ lấy kết quả của SSD.

#### 4. RetinaFace

**Đặc điểm:**

- SOTA (state-of-the-art), phát hiện cực kỳ chính xác, nhận diện tốt các khuôn mặt nhỏ, nghiêng, che khuất một phần.
- Chậm hơn Haar, nhưng cho kết quả tốt nhất với ảnh crop hoặc ảnh chất lượng cao.
  **Cách dùng trong hệ thống:**
  - Được thử như một backend trong DeepFace khi crop hoặc trích xuất embedding.
  - Nếu RetinaFace phát hiện tốt nhất, sẽ lấy kết quả của RetinaFace.

#### 5. Quy trình chọn backend detector trong DeepFace

- Khi cần trích xuất embedding hoặc crop khuôn mặt, hệ thống sẽ thử lần lượt các backend: Haar (opencv), RetinaFace, SSD, MTCNN.
- Nếu backend nào phát hiện được khuôn mặt với confidence cao nhất, sẽ lấy kết quả của backend đó.
- Điều này giúp tăng độ chính xác nhận diện, giảm false negative do từng thuật toán có ưu nhược điểm riêng.

#### 6. Ưu nhược điểm tổng hợp

| Thuật toán | Tốc độ   | Độ chính xác | Đa góc/ánh sáng | Phát hiện nhiều mặt | Căn chỉnh |
| ------------ | ----------- | ---------------- | ------------------- | ----------------------- | ----------- |
| Haar         | Rất nhanh  | Trung bình      | Kém                | Có                     | Không      |
| MTCNN        | Trung bình | Tốt             | Tốt                | Tốt                    | Có         |
| SSD          | Nhanh       | Tốt             | Tốt                | Tốt                    | Không      |
| RetinaFace   | Chậm       | Xuất sắc       | Xuất sắc          | Xuất sắc              | Có         |

### b. Cắt và lưu ảnh crop

**Quy trình:**

1. Thử lần lượt các backend detector (Haar, MTCNN, SSD, RetinaFace) để phát hiện khuôn mặt tốt nhất trên ảnh.
2. Lấy bounding box khuôn mặt có confidence cao nhất.
3. Cắt ảnh với margin (ví dụ 20px).
4. Kiểm tra kích thước crop (>= 50x50).
5. Lưu file crop vào uploads/face_crops/ với tên theo timestamp.
6. Kiểm tra lại file crop có tồn tại và đọc được không.

---

## 3. Phát hiện vật cản (Obstacle Detection)

### a. Background Subtraction (MOG2)

- **Quy trình:**
  1. Chuyển frame sang grayscale, GaussianBlur.
  2. Áp dụng MOG2 để lấy foreground mask.
  3. Morphology (open, close) để loại bỏ noise.
  4. Threshold để lấy vùng foreground rõ nét.
  5. Tìm contour, lọc theo diện tích (800–30000), aspect ratio (0.25–3.5).
  6. Tính solidity (độ đặc) của contour.
  7. Nếu solidity > 0.3, tính confidence dựa trên diện tích và solidity.

### b. Edge Density

- **Quy trình:**
  1. Chuyển grayscale, GaussianBlur.
  2. Tính gradient (Sobel X, Y), lấy magnitude.
  3. Chuẩn hóa, threshold để lấy vùng nhiều cạnh.
  4. Morphology (open, close).
  5. Tìm contour, lọc theo diện tích, aspect ratio.
  6. Tính mật độ cạnh trong bounding box.
  7. Nếu mật độ cạnh 0.1–0.8, tính confidence dựa trên mật độ và diện tích.

### c. Motion Detection (Optical Flow)

- **Quy trình:**
  1. Lưu frame trước (prev_gray).
  2. Tính optical flow giữa prev_gray và current_gray (Farneback).
  3. Tính magnitude, tạo mask vùng chuyển động mạnh (mag > 1.5).
  4. Morphology, tìm contour, lọc diện tích, aspect ratio.
  5. Tính motion intensity trong bounding box, tính confidence.

### d. Cascade Detection

- **Quy trình:**
  1. Dùng các cascade (fullbody, upperbody, lowerbody) để phát hiện vật thể lớn.
  2. Lọc theo diện tích, aspect ratio.

### e. ROI Filtering

- **Quy trình:**
  1. Chỉ giữ vật cản có tâm nằm trong vùng ROI (ví dụ 30–90% chiều cao khung hình).
  2. Tính distance_score dựa trên vị trí dọc.
  3. Cập nhật confidence dựa trên distance_score.

### f. Temporal Filtering

- **Quy trình:**
  1. Lưu số lượng vật cản qua 10 frame gần nhất (deque).
  2. Nếu >= 6/10 frame có vật cản, giữ lại.
  3. Kiểm tra vật cản xuất hiện liên tiếp qua nhiều frame (so sánh vị trí, khoảng cách < 40px).
  4. Nếu xuất hiện >= 2 lần liên tiếp, tăng consistency_score.

### g. Non-Maximum Suppression

- **Quy trình:**
  1. Sắp xếp các bounding box theo confidence giảm dần.
  2. Loại bỏ các box có overlap > 0.4 với box đã chọn.

### h. Danger Level

- **Công thức:**
  - Dựa trên: diện tích, vị trí dọc (càng thấp càng nguy hiểm), vị trí ngang (gần giữa), chạm biên khung hình, confidence.
  - Tính điểm nguy hiểm, phân loại: 0 (thấp), 1 (trung bình), 2 (cao).

---

## 4. Quản lý cache embedding

- **faces_cache:**
  - List các dict chứa embedding, tên, mối quan hệ, email, ảnh.
  - Được đồng bộ từ database mỗi 10 giây hoặc sau khi thêm/xóa.
  - Sử dụng threading.Lock để tránh race condition.

---

## 5. Đa luồng (Threading)

- **Camera Thread:**
  - Đọc frame liên tục từ RTSP, giảm độ trễ.
- **Recognition Thread:**
  - Xử lý nhận diện, lưu lịch sử, gửi thông báo.
- **Gửi thông báo (send_to_pi_async):**
  - Gửi HTTP POST tới Pi server bằng thread riêng, kiểm tra cooldown từng người.

---

## 6. Các hàm phụ trợ

- **extract_and_save_face_crop:**
  - Thử nhiều detector để cắt khuôn mặt tốt nhất.
  - Lưu file crop, kiểm tra lại file.
- **get_embedding_from_image_file:**
  - Trích xuất embedding từ file ảnh bất kỳ.
- **save_recognition_to_history:**
  - Lưu kết quả nhận diện vào history, fallback nếu lỗi.

---

## 7. Quản lý thiết bị, người dùng, vị trí

- **DeviceModel:**
  - Theo dõi trạng thái online/offline qua heartbeat, tự động chuyển offline nếu không có heartbeat >2 phút.
- **UserModel:**
  - Hash mật khẩu, xác thực đăng nhập, CRUD user.
- **LocationModel:**
  - Lưu vị trí, truy xuất lịch sử, xóa lịch sử.

---

## 8. Tối ưu hiệu năng

- **RAM cache embedding**: tăng tốc nhận diện.
- **Đa luồng**: giảm độ trễ, không block camera/nhận diện.
- **Lọc nhiễu không gian & thời gian**: giảm false positive.
- **Thử nhiều detector**: tăng khả năng phát hiện khuôn mặt/vật cản.

---

Nếu cần giải thích thuật toán cụ thể từng hàm, từng dòng code, hoặc ví dụ thực tế, hãy chỉ rõ module/hàm bạn muốn tìm hiểu.
