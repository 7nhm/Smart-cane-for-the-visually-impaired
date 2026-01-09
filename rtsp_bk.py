# rtsp_stream.py - FACE DETECTION + RECOGNITION + AUDIO ANNOUNCEMENT
import cv2
import threading
import time
import os
import numpy as np
import requests
import json
from datetime import datetime
import sys
import traceback
import collections

# Thêm đường dẫn để import face_model
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

print(f"Project root: {project_root}")

# ================================================================
PI4_BASE_URL = "http://100.93.7.42:5001"
PI4_API_TOKEN = "Bearer visionmate_pi_token"
RTSP_URL = "rtsp://100.93.7.42:8554/usb"

# Biến toàn cục
latest_frame = None
latest_frame_lock = threading.Lock()
stream_active = True
frame_ready = threading.Event()

# Biến theo dõi
current_people = {}
last_recognition_time = 0

# Cooldown phát âm thanh - THEO YÊU CẦU
UNKNOWN_COOLDOWN = 10      # Người lạ: 10 giây
KNOWN_COOLDOWN = 5       # Người quen: 5 giây cho cùng người
announce_history = {"unknown": 0, "known": {}}  # Lịch sử phát âm thanh

# ================================================================
# IMPORT FACE_MODEL AN TOÀN
# ================================================================
def import_face_model():
    """Import face_model một cách an toàn"""
    try:
        model_dir = os.path.join(project_root, "Model")
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        
        from face_model import recognize_face_simple
        print("Import face_model successful!")
        return recognize_face_simple
    except ImportError as e:
        print(f"Import error: {e}")
        return None
    except Exception as e:
        print(f"Other error: {e}")
        return None

# Import face_model function
recognize_face_func = import_face_model()

if recognize_face_func is None:
    print("⚠️ Cannot import face_model, creating fallback function")
    def recognize_face_simple_fallback(frame_path):
        return ("UNKNOWN", 0.0, "Unknown", None)
    recognize_face_func = recognize_face_simple_fallback

# ================================================================
# FACE DETECTION NÂNG CAO - GIẢM NHIỄU
# ================================================================
class FaceDetector:
    def __init__(self):
        self.face_cascade = None
        self.load_cascade()
        
        # Face tracking để giảm false positive
        self.face_positions = collections.deque(maxlen=8)  # Lưu 8 frame gần nhất
        self.confidence_frames = 0  # Đếm số frame có khuôn mặt liên tiếp
        self.last_face_time = 0
        self.current_face_id = 0
        
    def load_cascade(self):
        """Tải cascade classifier"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("Face cascade loaded")
        except:
            self.face_cascade = None
            print("Cannot load face cascade")
    
    def detect_reliable_face(self, frame):
        """
        Phát hiện khuôn mặt đáng tin cậy (tránh false positive)
        Trả về (x, y, w, h) hoặc None nếu không có khuôn mặt thật
        """
        if self.face_cascade is None or frame is None:
            self.confidence_frames = max(0, self.confidence_frames - 2)
            return None
        
        try:
            # Chuẩn bị frame
            height, width = frame.shape[:2]
            
            # Resize để tăng tốc độ
            if width > 400:
                scale = 400 / width
                small_w = 400
                small_h = int(height * scale)
                small_frame = cv2.resize(frame, (small_w, small_h))
            else:
                small_frame = frame
                scale = 1.0
            
            # Chuyển sang grayscale và xử lý
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Phát hiện khuôn mặt với tham số nghiêm ngặt hơn
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=4,        # Tăng để giảm false positive
                minSize=(60, 60),      # Kích thước tối thiểu
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                # Không có khuôn mặt
                self.confidence_frames = max(0, self.confidence_frames - 1)
                return None
            
            # Lấy khuôn mặt lớn nhất
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Scale lại tọa độ
            if scale != 1.0:
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
            # Kiểm tra kích thước
            if w < 70 or h < 70:  # Yêu cầu kích thước đủ lớn
                self.confidence_frames = max(0, self.confidence_frames - 1)
                return None
            
            # Kiểm tra tính ổn định
            current_pos = (x, y, w, h)
            self.face_positions.append(current_pos)
            
            # Tăng confidence nếu khuôn mặt ổn định
            if len(self.face_positions) >= 2:
                last_pos = self.face_positions[-2] if len(self.face_positions) >= 2 else None
                if last_pos:
                    # Tính khoảng cách di chuyển
                    last_x, last_y, last_w, last_h = last_pos
                    distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                    
                    # Nếu di chuyển ít (ổn định)
                    if distance < 50:  # pixel threshold
                        self.confidence_frames += 1
                    else:
                        self.confidence_frames = max(1, self.confidence_frames - 1)
            else:
                self.confidence_frames = 1
            
            # Chỉ trả về khuôn mặt nếu đã ổn định qua ít nhất 2 frame
            if self.confidence_frames >= 2:
                self.last_face_time = time.time()
                return (x, y, w, h)
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
            return None

# Khởi tạo face detector
face_detector = FaceDetector()

# ================================================================
# HÀM GỬI KẾT QUẢ VỀ PI 
# ================================================================
def send_to_pi_async(name, confidence):
    global announce_history
    
    current_time = time.time()
    name_str = str(name)
    
    # KIỂM TRA COOLDOWN THEO YÊU CẦU
    if name_str == "UNKNOWN":
        # Kiểm tra cooldown cho unknown
        if current_time - announce_history["unknown"] < UNKNOWN_COOLDOWN:
            print(f"[AUDIO] Unknown cooldown: {UNKNOWN_COOLDOWN - (current_time - announce_history['unknown']):.1f}s")
            return False
        
        # Unknown cần confidence tối thiểu
        if confidence < 0.5:
            print(f"[AUDIO] Unknown confidence too low: {confidence:.1%}")
            return False
            
        message = "Có người lạ"
        status = "unknown"
        announce_history["unknown"] = current_time
        
    else:
        # KIỂM TRA COOLDOWN CHO KNOWN - THEO TỪNG NGƯỜI
        if name_str in announce_history["known"]:
            last_announced = announce_history["known"][name_str]
            time_since_last = current_time - last_announced
            
            if time_since_last < KNOWN_COOLDOWN:
                print(f"⏳ [AUDIO] {name_str} cooldown: {KNOWN_COOLDOWN - time_since_last:.1f}s")
                return False
        
        # Known cần confidence cao hơn
        if confidence < 0.7:
            print(f"[AUDIO] {name_str} confidence too low: {confidence:.1%}")
            return False
            
        message = f"Xin chào {name_str}"
        status = "known"
        # CẬP NHẬT COOLDOWN CHO NGƯỜI NÀY (30 giây)
        announce_history["known"][name_str] = current_time
    
    print(f"[PI] Sending to audio: {name_str} ({confidence:.1%})")
    
    def send_request():
        payload = {
            "status": status,
            "name": name_str,
            "confidence": float(confidence),
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        headers = {
            "Authorization": PI4_API_TOKEN,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{PI4_BASE_URL}/api/receive_recognition",
                json=payload,
                headers=headers,
                timeout=2.0
            )
            
            if response.status_code == 200:
                print(f"[PI] Audio announcement sent for {name_str}")
            else:
                print(f"[PI] Error: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"[PI] Timeout")
        except Exception as e:
            print(f"[PI] Error: {e}")
    
    thread = threading.Thread(target=send_request, daemon=True)
    thread.start()
    
    return True

# ================================================================
# THREAD CAMERA - GIỮ NGUYÊN ĐỘ TRỄ THẤP
# ================================================================
def camera_thread():
    """Thread camera với độ trễ thấp"""
    global latest_frame, frame_ready
    
    print("📡 [CAM] Camera thread started")
    
    cap = None
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    
    # RTSP options đơn giản
    rtsp_url = f"{RTSP_URL}?tcp&buffer_size=1"
    
    print(f"📡 RTSP URL: {RTSP_URL}")
    
    while stream_active and reconnect_attempts < max_reconnect_attempts:
        try:
            if cap is None or not cap.isOpened():
                print(f"📡 Connecting {reconnect_attempts + 1}/{max_reconnect_attempts}")
                
                cap = cv2.VideoCapture(rtsp_url)
                
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 20)
                    
                    print("✅ Camera connected")
                    reconnect_attempts = 0
                else:
                    print("❌ Cannot open camera")
                    cap = None
                    reconnect_attempts += 1
                    time.sleep(2)
                    continue
            
            # Đọc frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("⚠️ Frame read failed")
                cap.release()
                cap = None
                reconnect_attempts += 1
                time.sleep(1)
                continue
            
            # Resize để giảm tải
            if frame.shape[1] > 640:
                height, width = frame.shape[:2]
                new_width = 640
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Cập nhật frame
            with latest_frame_lock:
                latest_frame = frame
            
            frame_ready.set()
            
            # Giới hạn FPS
            time.sleep(1/20)  # 20 FPS
            
        except Exception as e:
            print(f"❌ [CAM] Error: {str(e)[:100]}")
            if cap:
                cap.release()
                cap = None
            reconnect_attempts += 1
            time.sleep(1)
    
    if cap:
        cap.release()
    print("🛑 [CAM] Camera thread stopped")

# ================================================================
# 🧠 THREAD RECOGNITION - NHẬN DIỆN VÀ GỬI ÂM THANH
# ================================================================
def recognition_thread():
    """Thread nhận diện và gửi kết quả cho Pi phát âm thanh"""
    
    print("🧠 [RECOG] Recognition thread started")
    
    temp_dir = "uploads/rtsp_faces"
    os.makedirs(temp_dir, exist_ok=True)
    
    last_process_time = 0
    process_interval = 3  # Xử lý mỗi 3 giây
    
    while stream_active:
        try:
            current_time = time.time()
            
            # Cooldown cơ bản
            if current_time < last_process_time + process_interval:
                time.sleep(0.5)
                continue
            
            # Đợi frame mới
            if not frame_ready.wait(timeout=1.0):
                continue
            
            frame_ready.clear()
            
            # Lấy frame
            with latest_frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
            
            print("🔍 [FACE] Checking for reliable face...")
            
            # Phát hiện khuôn mặt đáng tin cậy
            face_rect = face_detector.detect_reliable_face(frame)
            
            if face_rect is None:
                print("[FACE] No reliable face detected")
                last_process_time = current_time
                continue
            
            x, y, w, h = face_rect
            print(f"[FACE] Reliable face detected: {w}x{h}")
            
            # Cắt khuôn mặt với margin
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            
            # Kiểm tra chất lượng crop
            if (face_crop.size == 0 or 
                face_crop.shape[0] < 50 or 
                face_crop.shape[1] < 50):
                print("[FACE] Face crop too small")
                continue
            
            # Lưu face crop
            timestamp = int(time.time() * 1000)
            face_path = os.path.join(temp_dir, f"face_{timestamp}.jpg")
            
            try:
                # Lưu với chất lượng vừa
                cv2.imwrite(face_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
                
                if not os.path.exists(face_path):
                    print("[FACE] Cannot save face")
                    continue
                
                file_size = os.path.getsize(face_path)
                if file_size < 1500:  # File quá nhỏ
                    print(f"[FACE] Face image too small: {file_size} bytes")
                    continue
                
                print(f"[FACE] Saved: {face_crop.shape} ({file_size} bytes)")
                
                # Nhận diện khuôn mặt
                print("Recognizing face...")
                recog_start = time.time()
                result = recognize_face_func(face_path)
                recog_time = time.time() - recog_start
                
                # Xử lý kết quả
                recognized_name = "UNKNOWN"
                confidence = 0.0
                
                if result and isinstance(result, (list, tuple)) and len(result) >= 2:
                    name, conf = result[0], result[1]
                    
                    # Xử lý tên
                    if not name or name in ["None", "NO_FACE", "NO_EMBEDDING", "", "unknown"]:
                        recognized_name = "UNKNOWN"
                    else:
                        recognized_name = str(name).strip()
                    
                    # Xử lý confidence
                    try:
                        confidence = float(conf) if conf else 0.0
                    except:
                        confidence = 0.0
                    
                    print(f"[RESULT] {recognized_name} ({confidence:.1%}) in {recog_time:.2f}s")
                    
                    # GỬI KẾT QUẢ CHO PI PHÁT ÂM THANH
                    send_to_pi_async(recognized_name, confidence)
                    
                    # Cập nhật tracking
                    current_people[recognized_name] = current_time
                else:
                    print(f"Invalid result format: {result}")
                    # Gửi unknown nếu không có kết quả
                    send_to_pi_async("UNKNOWN", 0.5)
                
                # Cập nhật thời gian
                last_process_time = current_time
                
            except Exception as e:
                print(f"[RECOG] Error: {e}")
                traceback.print_exc()
            finally:
                # Xóa file tạm
                if os.path.exists(face_path):
                    try:
                        os.remove(face_path)
                    except:
                        pass
            
            # Dọn dẹp tracking
            cleanup_old_people(60)  # 1 phút
            
        except Exception as e:
            print(f"[RECOG] Thread error: {e}")
            time.sleep(1)
    
    print("[RECOG] Recognition thread stopped")

def cleanup_old_people(max_age=60):
    """Dọn dẹp người cũ khỏi tracking"""
    try:
        current_time = time.time()
        to_remove = []
        
        for name, last_seen in list(current_people.items()):
            if current_time - last_seen > max_age:
                to_remove.append(name)
        
        for name in to_remove:
            del current_people[name]
        
        if to_remove:
            print(f"🗑️ Cleaned {len(to_remove)} old entries")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

# ================================================================
# STREAM CHO WEB - VẼ BOX KHI CÓ KHUÔN MẶT
# ================================================================
def rtsp_stream():
    """Stream cho web với box khi có khuôn mặt"""
    
    print("[STREAM] Starting web stream with face boxes")
    
    last_display_frame = None
    fps_counter = 0
    last_fps_time = time.time()
    frame_counter = 0
    
    # Biến để vẽ box 
    current_boxes = []
    box_alpha = 1.0
    box_fade_time = 0
    
    while stream_active:
        try:
            frame_counter += 1
            
            # Lấy frame
            with latest_frame_lock:
                if latest_frame is not None:
                    display_frame = latest_frame.copy()
                    last_display_frame = display_frame.copy()
                elif last_display_frame is not None:
                    display_frame = last_display_frame.copy()
                else:
                    # Frame trống
                    display_frame = np.zeros((360, 480, 3), dtype=np.uint8)
            
            # VẼ FACE BOXES KHI CÓ KHUÔN MẶT
            
            # Phát hiện khuôn mặt nhanh cho display (mỗi 3 frame)
            if frame_counter % 3 == 0:
                try:
                    # Phát hiện khuôn mặt đơn giản cho display
                    if face_detector.face_cascade is not None:
                        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        
                        faces = face_detector.face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=3,
                            minSize=(50, 50)
                        )
                        
                        if len(faces) > 0:
                            # Lấy khuôn mặt lớn nhất
                            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                            current_boxes = faces[:1]  # Chỉ lấy 1 khuôn mặt lớn nhất
                            box_fade_time = time.time()
                        else:
                            # Fade out boxes khi không có mặt
                            if time.time() - box_fade_time > 0.5:
                                current_boxes = []
                except:
                    current_boxes = []
            
            # Vẽ boxes
            for (x, y, w, h) in current_boxes:
                # Màu box (xanh lá cho face detected)
                color = (0, 255, 0)
                thickness = 2
                
                # Vẽ box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                
                # Vẽ label
                label = "FACE DETECTED"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Background cho label
                cv2.rectangle(
                    display_frame,
                    (x, y - label_height - 5),
                    (x + label_width, y),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    display_frame,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),  # Màu đen cho chữ
                    1
                )
            
            # Vẽ overlay thông tin
            people_count = len(current_people)
            
            # Status bar
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), 
                         (0, 0, 0, 180), -1)
            
            # Tiêu đề
            cv2.putText(display_frame, "🎥 VISIONMATE LIVE", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)
            
            # Thông tin nhận diện
            if len(current_boxes) > 0:
                status_text = "Face Detected"
                cv2.putText(display_frame, status_text, 
                           (display_frame.shape[1] - 120, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Thời gian
            time_text = datetime.now().strftime("%H:%M:%S")
            cv2.putText(display_frame, time_text, 
                       (display_frame.shape[1] - 80, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Thông tin người đã nhận diện
            if people_count > 0:
                info_y = display_frame.shape[0] - 10
                info_text = f"Recognized: {people_count}"
                cv2.putText(display_frame, info_text, 
                           (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            # FPS
            fps_counter += 1
            if time.time() - last_fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_fps_time = time.time()
                
                # Hiển thị FPS
                fps_color = (100, 255, 100) if fps >= 15 else (100, 100, 255)
                cv2.putText(display_frame, f"{fps}fps", 
                           (display_frame.shape[1] - 50, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, fps_color, 1)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', display_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 60,
            ])
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Giới hạn FPS để mượt
            time.sleep(1/25)  # 25 FPS
            
        except Exception as e:
            print(f"[STREAM] Error: {e}")
            time.sleep(0.1)
    
    print("[STREAM] Stream stopped")

# ================================================================
# HÀM KHỞI ĐỘNG
# ================================================================
def start_rtsp_system():
    """Khởi động hệ thống"""
    
    print("\n" + "="*60)
    print("STARTING RTSP SYSTEM - FACE RECOGNITION + AUDIO")
    print("="*60)
    print(f"Camera: {RTSP_URL}")
    print(f"Pi Server: {PI4_BASE_URL}")
    print(f"Face Detection: RELIABLE MODE")
    print(f"Display: Face boxes enabled")
    print(f"Audio Cooldown:")
    print(f"   - Unknown: {UNKNOWN_COOLDOWN}s")
    print(f"   - Known: {KNOWN_COOLDOWN}s per person")
    print("="*60)
    
    # Khởi động camera thread
    cam_thread = threading.Thread(target=camera_thread, daemon=True, name="CameraThread")
    cam_thread.start()
    time.sleep(1)
    
    # Khởi động recognition thread
    recog_thread = threading.Thread(target=recognition_thread, daemon=True, name="RecognitionThread")
    recog_thread.start()
    
    print("All threads started")
    print("System ready! (Face boxes + Audio announcement)")
    print("="*60 + "\n")

def stop_rtsp_system():
    """Dừng hệ thống"""
    global stream_active
    
    print("\nStopping RTSP system...")
    stream_active = False
    time.sleep(1)
    print("System stopped")

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("RTSP SYSTEM - FACE RECOGNITION + AUDIO ANNOUNCEMENT")
    
    start_rtsp_system()
    
    try:
        status_counter = 0
        
        while True:
            status_counter += 1
            
            # Hiển thị status mỗi 15 giây
            if status_counter % 15 == 0:
                people_list = list(current_people.keys())
                if len(people_list) > 4:
                    display_list = people_list[:4]
                    display_list.append("...")
                else:
                    display_list = people_list
                
                print(f"\n📊 STATUS: People={len(current_people)} | "
                      f"Recent: {display_list}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_rtsp_system()