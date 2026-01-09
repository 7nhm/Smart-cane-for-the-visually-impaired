"""
RTSP Stream System - Face Recognition + Obstacle Detection
Using two models: face_model.py and advanced_obstacle_detector.py
"""

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

# Thêm đường dẫn để import các model
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

print(f"Project root: {project_root}")

# ================================================================
# CONFIGURATION
# ================================================================
PI4_BASE_URL = "http://100.93.7.42:5001"
PI4_API_TOKEN = "Bearer visionmate_pi_token"
RTSP_URL = "rtsp://100.93.7.42:8554/usb"

# ================================================================
# IMPORT MODELS
# ================================================================

# 1. Import Advanced Obstacle Detector
try:
    model_dir = os.path.join(project_root, "Model")
    sys.path.insert(0, model_dir)
    from advanced_obstacle_detector import advanced_detector
    OBSTACLE_DETECTION_ENABLED = True
    print("Advanced Obstacle Detection enabled")
except Exception as e:
    OBSTACLE_DETECTION_ENABLED = False
    print(f"Obstacle detection disabled: {e}")
    advanced_detector = None

# 2. Import Face Model
try:
    model_dir = os.path.join(project_root, "Model")
    sys.path.insert(0, model_dir)
    from face_model import recognize_face_from_frame
    FACE_RECOGNITION_ENABLED = True
    print("Face Recognition enabled")
except Exception as e:
    FACE_RECOGNITION_ENABLED = False
    print(f"Face recognition disabled: {e}")
    recognize_face_from_frame = None

# ================================================================
# BIẾN TOÀN CỤC
# ================================================================
latest_frame = None
latest_frame_lock = threading.Lock()
stream_active = True
camera_connected = False

# Biến theo dõi
current_people = {}
obstacle_detections = []
face_boxes = []  # Lưu box khuôn mặt
obstacle_boxes = []  # Lưu box vật cản
last_face_result = ("UNKNOWN", 0.0, "Unknown", None)
last_face_detection_time = 0
last_obstacle_detection_time = 0

# Cooldown phát âm thanh
UNKNOWN_COOLDOWN = 8        # 8 giây
KNOWN_COOLDOWN = 5          # 5 giây
OBSTACLE_COOLDOWN = 15      # 15 giây
announce_history = {"unknown": 0, "known": {}, "obstacle": 0}

# ================================================================
# CONFIGURATION ĐƠN GIẢN
# ================================================================
# Cấu hình nhận diện khuôn mặt
FACE_CONFIDENCE_THRESHOLD = 0.65      # 65%
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80
FACE_AREA_MIN = 4000

# Cấu hình vật cản
OBSTACLE_CONFIDENCE_THRESHOLD = 0.65   # Confidence

# Biến ưu tiên âm thanh
last_face_audio_time = 0
AUDIO_PRIORITY_TIMEOUT = 5.0   # Sau khi phát âm thanh mặt, chờ 5s mới phát vật cản

# ================================================================
# FACE DETECTOR ĐƠN GIẢN
# ================================================================
class SimpleFaceDetector:
    def __init__(self):
        self.face_cascade = None
        self.load_cascade()
        
    def load_cascade(self):
        """Tải cascade classifier"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if not self.face_cascade.empty():
                print("Simple Face cascade loaded")
            else:
                self.face_cascade = None
                print("Cannot load face cascade")
        except Exception as e:
            self.face_cascade = None
            print(f"Cascade load error: {e}")
    
    def detect_faces_simple(self, frame):
        """Phát hiện khuôn mặt đơn giản"""
        if self.face_cascade is None or frame is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            valid_faces = []
            for (x, y, w, h) in faces:
                if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
                    continue
                
                face_area = w * h
                if face_area < FACE_AREA_MIN:
                    continue
                
                aspect_ratio = w / h
                if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                    continue
                
                valid_faces.append((x, y, w, h))
            
            return valid_faces
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

# Khởi tạo face detector
face_detector = SimpleFaceDetector()

# ================================================================
# HÀM KIỂM TRA ƯU TIÊN ÂM THANH
# ================================================================
def check_audio_priority():
    """Kiểm tra ưu tiên âm thanh"""
    global last_face_audio_time
    
    current_time = time.time()
    
    # Nếu vừa phát âm thanh mặt
    if (current_time - last_face_audio_time) < AUDIO_PRIORITY_TIMEOUT:
        return "FACE_PRIORITY"
    
    return "OBSTACLE_ALLOWED"

def update_face_audio_time():
    """Cập nhật thời gian phát âm thanh mặt"""
    global last_face_audio_time
    last_face_audio_time = time.time()
    print(f"[AUDIO PRIORITY] Face audio at {last_face_audio_time}")

# ================================================================
# HÀM VẼ BOXES
# ================================================================
def draw_face_boxes(frame, boxes):
    """Vẽ box khuôn mặt"""
    if len(boxes) == 0:
        return frame
    
    for i, (x, y, w, h) in enumerate(boxes):
        color = (0, 255, 0)  # Xanh lá
        thickness = 2
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        label = f"FACE"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(frame, 
                     (x, y - label_h - 5), 
                     (x + label_w, y), 
                     color, -1)
        
        cv2.putText(frame, label, 
                   (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 0, 0), 1)
    
    return frame

def draw_obstacle_boxes(frame, obstacles):
    """Vẽ box vật cản"""
    if len(obstacles) == 0:
        return frame
    
    for obstacle in obstacles:
        try:
            if 'bbox' not in obstacle:
                continue
                
            x, y, w, h = obstacle['bbox']
            confidence = obstacle.get('confidence', 0)
            
            # Màu dựa trên confidence
            if confidence > 0.8:
                color = (0, 0, 255)      # Đỏ - rất cao
                thickness = 3
            elif confidence > 0.7:
                color = (0, 140, 255)    # Cam - cao
                thickness = 2
            else:
                color = (0, 255, 255)    # Vàng - trung bình
                thickness = 1
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            label_text = f"OBSTACLE: {confidence:.1%}"
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            cv2.rectangle(frame, 
                        (x, y - label_h - 2), 
                        (x + label_w, y), 
                        color, -1)
            
            cv2.putText(frame, label_text, 
                       (x, y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       (255, 255, 255), 1)
                              
        except Exception as e:
            continue
    
    return frame

def draw_overlay_info(frame):
    """Vẽ overlay thông tin"""
    height, width = frame.shape[:2]
    
    status_color = (0, 255, 0) if camera_connected else (0, 0, 255)
    status_text = "LIVE" if camera_connected else "OFFLINE"
    
    cv2.rectangle(frame, (0, 0), (width, 35), (0, 0, 0, 180), -1)
    cv2.putText(frame, f"{status_text}", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    face_count = len(face_boxes)
    face_text = f"Faces: {face_count}"
    cv2.putText(frame, face_text, 
                (width - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    if OBSTACLE_DETECTION_ENABLED:
        obstacle_count = len(obstacle_boxes)
        obs_color = (100, 100, 255) if obstacle_count > 0 else (200, 200, 200)
        obs_text = f"Obstacles: {obstacle_count}"
        cv2.putText(frame, obs_text, 
                    (width - 130, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obs_color, 1)
    
    time_text = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, time_text, 
                (width - 100, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    priority_status = check_audio_priority()
    priority_color = (0, 255, 0) if priority_status == "FACE_PRIORITY" else (255, 255, 0)
    priority_text = f"Audio: {'FACE' if priority_status == 'FACE_PRIORITY' else 'FREE'}"
    cv2.putText(frame, priority_text, 
                (10, height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, priority_color, 1)
    
    if last_face_result[0] != "UNKNOWN" and time.time() - last_face_detection_time < 5:
        name, conf, _, _ = last_face_result
        result_text = f"Last: {name} ({conf:.1%})"
        cv2.putText(frame, result_text, 
                    (width - 150, height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return frame

# ================================================================
# HÀM GỬI KẾT QUẢ VỀ PI
# ================================================================
def send_to_pi_async(name, confidence, is_obstacle=False, danger_level=0):
    global announce_history, last_face_audio_time
    
    current_time = time.time()
    name_str = str(name)
    
    if hasattr(confidence, 'item'):
        confidence = float(confidence.item())
    else:
        confidence = float(confidence)
    
    if is_obstacle:
        # Cooldown check
        if current_time - announce_history["obstacle"] < OBSTACLE_COOLDOWN:
            return False
        
        # Ưu tiên âm thanh mặt
        audio_priority = check_audio_priority()
        if audio_priority == "FACE_PRIORITY":
            return False
        
        # Confidence threshold
        if confidence < OBSTACLE_CONFIDENCE_THRESHOLD:
            return False
        
        message = "Cẩn thận! Có vật cản phía trước."
        status = "obstacle"
        announce_history["obstacle"] = current_time
        
        print(f"[PI] OBSTACLE ALERT: {message} (confidence: {confidence:.1%})")
        
    else:
        # Face recognition
        if name_str == "UNKNOWN":
            if current_time - announce_history["unknown"] < UNKNOWN_COOLDOWN:
                return False
            
            if confidence < FACE_CONFIDENCE_THRESHOLD:
                return False
                
            message = "Có người lạ"
            status = "unknown"
            announce_history["unknown"] = current_time
            print(f"[PI] UNKNOWN: {message} (conf: {confidence:.1%})")
            
            update_face_audio_time()
            
        else:
            if name_str in announce_history["known"]:
                last_announced = announce_history["known"][name_str]
                if current_time - last_announced < KNOWN_COOLDOWN:
                    return False
            
            if confidence < 0.70:
                return False
            
            message = f"Đây là {name_str}"
            status = "known"
            announce_history["known"][name_str] = current_time
            print(f"[PI] KNOWN: {name_str} (conf: {confidence:.1%})")
            
            update_face_audio_time()
    
    # Gửi request
    def send_request():
        payload = {
            "status": status,
            "name": name_str,
            "confidence": confidence,
            "message": message,
            "is_obstacle": is_obstacle,
            "danger_level": int(danger_level),
            "timestamp": datetime.now().isoformat(),
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
                print(f"[PI] Audio sent: {message}")
                
        except Exception as e:
            print(f"[PI] Send error: {e}")
    
    thread = threading.Thread(target=send_request, daemon=True)
    thread.start()
    
    return True

# ================================================================
# THREAD CAMERA
# ================================================================
def camera_thread():
    """Thread camera"""
    global latest_frame, camera_connected
    
    print("[CAM] Camera thread started")
    
    cap = None
    
    while stream_active:
        try:
            if cap is None or not cap.isOpened():
                print(f"Connecting to RTSP...")
                
                cap = cv2.VideoCapture(RTSP_URL)
                
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 20)
                    
                    print("Camera connected")
                    camera_connected = True
                else:
                    camera_connected = False
                    time.sleep(2)
                    continue
            
            ret, frame = cap.read()
            
            if not ret or frame is None:
                camera_connected = False
                cap.release()
                cap = None
                time.sleep(1)
                continue
            
            if frame.shape[1] > 640:
                height, width = frame.shape[:2]
                new_width = 640
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))
            
            with latest_frame_lock:
                latest_frame = frame
            
            camera_connected = True
            
            time.sleep(1/20)
            
        except Exception as e:
            camera_connected = False
            if cap:
                cap.release()
                cap = None
            time.sleep(1)
    
    if cap:
        cap.release()
    print("[CAM] Camera thread stopped")

# ================================================================
# THREAD FACE RECOGNITION
# ================================================================
def face_recognition_thread():
    """Thread nhận diện khuôn mặt"""
    global last_face_result, last_face_detection_time, current_people, face_boxes
    
    print("[FACE] Face recognition thread started")
    
    temp_dir = "uploads/rtsp_faces"
    os.makedirs(temp_dir, exist_ok=True)
    
    last_process_time = 0
    process_interval = 2.0  # Xử lý mỗi 2 giây
    
    while stream_active:
        try:
            current_time = time.time()
            
            if current_time - last_process_time < process_interval:
                time.sleep(0.2)
                continue
            
            if not camera_connected:
                time.sleep(1)
                continue
            
            with latest_frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
            
            # Phát hiện khuôn mặt
            boxes = face_detector.detect_faces_simple(frame)
            face_boxes = boxes
            
            if len(boxes) == 0:
                last_process_time = current_time
                continue
            
            print(f"[FACE] Found {len(boxes)} faces")
            
            # Chỉ xử lý mặt đầu tiên
            x, y, w, h = boxes[0]
            
            margin = 15
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0 or face_crop.shape[0] < 70 or face_crop.shape[1] < 70:
                continue
            
            # Lưu ảnh tạm để nhận diện
            timestamp = int(time.time() * 1000)
            face_path = os.path.join(temp_dir, f"face_{timestamp}.jpg")
            
            try:
                cv2.imwrite(face_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                if not os.path.exists(face_path):
                    continue
                
                # Gọi hàm nhận diện từ face_model.py
                if FACE_RECOGNITION_ENABLED and recognize_face_from_frame:
                    result = recognize_face_from_frame(face_path, auto_save=True)
                    
                    if result and isinstance(result, (list, tuple)) and len(result) >= 2:
                        name, conf, relationship, _ = result
                        
                        if not name or name in ["None", "", "unknown"]:
                            recognized_name = "UNKNOWN"
                        else:
                            recognized_name = str(name).strip()
                        
                        try:
                            confidence = float(conf) if conf else 0.0
                        except:
                            confidence = 0.0
                        
                        print(f"[FACE RESULT] {recognized_name} ({confidence:.1%})")
                        
                        # Gửi về PI nếu đạt ngưỡng
                        if recognized_name == "UNKNOWN" and confidence >= FACE_CONFIDENCE_THRESHOLD:
                            send_to_pi_async(recognized_name, confidence, is_obstacle=False)
                            last_face_result = (recognized_name, confidence, relationship, face_crop)
                            last_face_detection_time = current_time
                        elif recognized_name != "UNKNOWN" and confidence >= 0.70:
                            send_to_pi_async(recognized_name, confidence, is_obstacle=False)
                            current_people[recognized_name] = current_time
                            last_face_result = (recognized_name, confidence, relationship, face_crop)
                            last_face_detection_time = current_time
                
                last_process_time = current_time
                
            except Exception as e:
                print(f"[FACE] Error: {e}")
            finally:
                # Xóa file tạm
                if os.path.exists(face_path):
                    try:
                        os.remove(face_path)
                    except:
                        pass
            
            # Dọn dẹp tracking cũ
            if current_time % 30 < 0.1:
                cleanup_old_people(60)
            
        except Exception as e:
            print(f"[FACE] Thread error: {e}")
            traceback.print_exc()
            time.sleep(1)
    
    print("[FACE] Recognition thread stopped")

# ================================================================
# THREAD OBSTACLE DETECTION
# ================================================================
def obstacle_detection_thread():
    """Thread phát hiện vật cản"""
    global obstacle_detections, obstacle_boxes, last_obstacle_detection_time
    
    print("[OBSTACLE] Obstacle detection thread started")
    
    last_process_time = 0
    process_interval = 3.0  # Xử lý mỗi 3 giây
    
    while stream_active:
        try:
            current_time = time.time()
            
            if current_time - last_process_time < process_interval:
                time.sleep(0.5)
                continue
            
            if not camera_connected:
                time.sleep(1)
                continue
            
            with latest_frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
            
            if not OBSTACLE_DETECTION_ENABLED or advanced_detector is None:
                time.sleep(2)
                continue
            
            # Gọi hàm detect từ advanced_obstacle_detector.py
            has_obstacle, obstacles, confidence, danger_level = advanced_detector.detect_obstacles(frame)
            
            if hasattr(confidence, 'item'):
                confidence = float(confidence.item())
            else:
                confidence = float(confidence)
            
            filtered_obstacles = []
            
            if has_obstacle:
                for obstacle in obstacles:
                    if 'bbox' not in obstacle:
                        continue
                    
                    obs_confidence = obstacle.get('confidence', 0.5)
                    if obs_confidence < OBSTACLE_CONFIDENCE_THRESHOLD:
                        continue
                    
                    filtered_obstacles.append(obstacle)
            
            obstacle_detections = filtered_obstacles
            obstacle_boxes = filtered_obstacles.copy()
            
            if len(filtered_obstacles) > 0:
                print(f"[OBSTACLE] Found {len(filtered_obstacles)} obstacles")
                
                # Kiểm tra xem có nên gửi cảnh báo không
                if len(filtered_obstacles) > 0:
                    audio_priority = check_audio_priority()
                    
                    if audio_priority == "OBSTACLE_ALLOWED":
                        # Gửi cảnh báo vật cản
                        send_to_pi_async(
                            "OBSTACLE", 
                            confidence, 
                            is_obstacle=True, 
                            danger_level=int(danger_level)
                        )
                    else:
                        print(f"[AUDIO PRIORITY] Blocking obstacle - Face priority")
            
            last_obstacle_detection_time = current_time
            last_process_time = current_time
            
        except Exception as e:
            print(f"[OBSTACLE] Error: {e}")
            traceback.print_exc()
            time.sleep(2)
    
    print("[OBSTACLE] Thread stopped")

def cleanup_old_people(max_age=60):
    """Dọn dẹp tracking"""
    global current_people
    
    try:
        current_time = time.time()
        to_remove = []
        
        for name, last_seen in list(current_people.items()):
            if current_time - last_seen > max_age:
                to_remove.append(name)
        
        for name in to_remove:
            del current_people[name]
        
        if to_remove:
            print(f"Cleaned {len(to_remove)} old entries")
    except Exception as e:
        print(f"Cleanup error: {e}")

# ================================================================
# STREAM CHO WEB
# ================================================================
def rtsp_stream():
    """Stream cho web"""
    
    print("[WEB_STREAM] Starting stream")
    
    fallback_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(fallback_frame, "NO CAMERA", (220, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    fps_counter = 0
    last_fps_time = time.time()
    last_valid_frame = None
    last_display_update = 0
    
    while stream_active:
        try:
            fps_counter += 1
            
            display_frame = None
            
            with latest_frame_lock:
                if latest_frame is not None:
                    display_frame = latest_frame.copy()
                    last_valid_frame = display_frame.copy()
                elif last_valid_frame is not None:
                    display_frame = last_valid_frame.copy()
                else:
                    display_frame = fallback_frame.copy()
            
            current_time = time.time()
            
            # Cập nhật phát hiện khuôn mặt để hiển thị
            if camera_connected and display_frame is not None:
                if current_time - last_display_update > 0.5:
                    boxes = face_detector.detect_faces_simple(display_frame)
                    face_boxes = boxes
                    last_display_update = current_time
            
            # Vẽ overlay
            if camera_connected and display_frame is not None:
                if face_boxes:
                    display_frame = draw_face_boxes(display_frame, face_boxes)
                
                if obstacle_boxes:
                    display_frame = draw_obstacle_boxes(display_frame, obstacle_boxes)
            
            display_frame = draw_overlay_info(display_frame)
            
            # Hiển thị FPS
            if time.time() - last_fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_fps_time = time.time()
                
                fps_color = (100, 255, 100) if fps >= 15 else (100, 100, 255)
                cv2.putText(display_frame, f"FPS: {fps}", 
                           (display_frame.shape[1] - 100, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', display_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 80
            ])
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(1/20)
            
        except Exception as e:
            print(f"[STREAM] Error: {e}")
            ret, buffer = cv2.imencode('.jpg', fallback_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    
    print("[STREAM] Stream stopped")

# ================================================================
# HÀM KHỞI ĐỘNG HỆ THỐNG
# ================================================================
def start_rtsp_system():
    """Khởi động hệ thống"""
    
    print("\n" + "="*60)
    print("RTSP SYSTEM - FACE & OBSTACLE DETECTION")
    print("="*60)
    print(f"Camera: {RTSP_URL}")
    print(f"Face Recognition: {'ENABLED' if FACE_RECOGNITION_ENABLED else 'DISABLED'}")
    print(f"  - Min confidence: {FACE_CONFIDENCE_THRESHOLD:.0%}")
    print(f"Obstacle Detection: {'ENABLED' if OBSTACLE_DETECTION_ENABLED else 'DISABLED'}")
    print(f"  - Min confidence: {OBSTACLE_CONFIDENCE_THRESHOLD:.0%}")
    print(f"Audio Cooldowns:")
    print(f"  - Unknown face: {UNKNOWN_COOLDOWN}s")
    print(f"  - Known face: {KNOWN_COOLDOWN}s")
    print(f"  - Obstacle: {OBSTACLE_COOLDOWN}s")
    print("="*60)
    
    # Khởi động threads
    cam_thread = threading.Thread(target=camera_thread, daemon=True, name="CameraThread")
    cam_thread.start()
    time.sleep(2)
    
    if FACE_RECOGNITION_ENABLED:
        face_thread = threading.Thread(target=face_recognition_thread, daemon=True, name="FaceThread")
        face_thread.start()
        print("Face recognition started")
    
    if OBSTACLE_DETECTION_ENABLED and advanced_detector is not None:
        obstacle_thread = threading.Thread(target=obstacle_detection_thread, daemon=True, name="ObstacleThread")
        obstacle_thread.start()
        print("Obstacle detection started")
    
    print("All threads started")
    print("System running...")
    print("="*60 + "\n")

def stop_rtsp_system():
    """Dừng hệ thống"""
    global stream_active
    
    print("\nStopping RTSP system...")
    stream_active = False
    time.sleep(2)
    print("System stopped")

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("RTSP SYSTEM - FACE & OBSTACLE DETECTION")
    
    start_rtsp_system()
    
    try:
        while True:
            time.sleep(10)
            
            print(f"\nSYSTEM STATUS:")
            print(f"  Camera: {'Connected' if camera_connected else 'Disconnected'}")
            print(f"  Face Tracking: {len(current_people)} people")
            
            if len(current_people) > 0:
                recent = list(current_people.keys())[:3]
                print(f"  Recent Faces: {recent}")
            
            print(f"  Face Boxes: {len(face_boxes)}")
            print(f"  Obstacles: {len(obstacle_boxes)}")
            
            priority_status = check_audio_priority()
            print(f"  Audio Priority: {'FACE' if priority_status == 'FACE_PRIORITY' else 'FREE'}")
            print(f"  Time since last face audio: {time.time() - last_face_audio_time:.1f}s")
            
            if last_face_result[0] != "UNKNOWN":
                print(f"  Last face: {last_face_result[0]} ({last_face_result[1]:.1%})")
            
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        stop_rtsp_system()