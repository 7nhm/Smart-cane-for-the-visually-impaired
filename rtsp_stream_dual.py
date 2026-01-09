"""
RTSP Stream System - Face Recognition + Obstacle Detection
SIMPLIFIED: No distance calculation, just detect large obstacles
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
import collections

# Thêm đường dẫn để import face_model
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

# Import Advanced Obstacle Detector
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

# ================================================================
# BIẾN TOÀN CỤC
# ================================================================
latest_frame = None
latest_frame_lock = threading.Lock()
stream_active = True
frame_ready = threading.Event()
camera_connected = False

# Biến theo dõi
current_people = {}
obstacle_detections = []
face_boxes = []  # Lưu box khuôn mặt
obstacle_boxes = []  # Lưu box vật cản
last_face_result = ("UNKNOWN", 0.0, None)
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
FACE_CONFIDENCE_THRESHOLD = 0.7      # 70%
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80
FACE_AREA_MIN = 4000
MAX_FACES = 1
FACE_STABILITY_THRESHOLD = 3

# Cấu hình vật cản ĐƠN GIẢN: CHỈ phát hiện vật LỚN (gần khung hình)
OBSTACLE_MIN_RATIO = 0.15      # Vật phải chiếm ít nhất 15% khung hình
OBSTACLE_MAX_RATIO = 0.60      # Vật chiếm tối đa 60% (tránh cảnh báo khi camera bị che)
OBSTACLE_CONFIDENCE_THRESHOLD = 0.7   # Confidence cao
OBSTACLE_MIN_AREA = 5000       # Diện tích tối thiểu

# Biến ưu tiên âm thanh
last_face_audio_time = 0
AUDIO_PRIORITY_TIMEOUT = 5.0   # Sau khi phát âm thanh mặt, chờ 5s mới phát vật cản

# Filter tracking
face_stability_counter = 0

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
        global face_stability_counter
        
        if self.face_cascade is None or frame is None:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            # Resize để tăng tốc
            if width > 400:
                scale = 400 / width
                small_w = 400
                small_h = int(height * scale)
                small_frame = cv2.resize(frame, (small_w, small_h))
            else:
                small_frame = frame
                scale = 1.0
            
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                face_stability_counter = max(0, face_stability_counter - 1)
                return []
            
            valid_faces = []
            for (x, y, w, h) in faces:
                if scale != 1.0:
                    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                
                if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
                    continue
                
                face_area = w * h
                if face_area < FACE_AREA_MIN:
                    continue
                
                aspect_ratio = w / h
                if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                    continue
                
                valid_faces.append((x, y, w, h))
            
            if len(valid_faces) == 0:
                face_stability_counter = max(0, face_stability_counter - 1)
                return []
            
            # Tăng stability khi phát hiện mặt
            face_stability_counter = min(FACE_STABILITY_THRESHOLD, face_stability_counter + 1)
            
            return valid_faces if face_stability_counter >= FACE_STABILITY_THRESHOLD else []
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

# Khởi tạo face detector
face_detector = SimpleFaceDetector()

# ================================================================
# IMPORT FACE_MODEL
# ================================================================
def import_face_model():
    """Import face_model"""
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
    print("Cannot import face_model, creating fallback function")
    def recognize_face_simple_fallback(frame_path):
        return ("UNKNOWN", 0.0, "Unknown", None)
    recognize_face_func = recognize_face_simple_fallback

# ================================================================
# HÀM KIỂM TRA VẬT CẢN LỚN (GẦN)
# ================================================================
def is_large_obstacle(bbox, frame_width, frame_height):
    """
    Kiểm tra xem vật cản có LỚN (gần khung hình) không
    Dựa trên tỉ lệ diện tích so với khung hình
    """
    x, y, w, h = bbox
    area = w * h
    frame_area = frame_width * frame_height
    
    # Tính tỉ lệ diện tích
    area_ratio = area / frame_area
    
    # Vật cản LỚN nếu chiếm từ 15% đến 60% khung hình
    # - Dưới 15%: vật nhỏ/ở xa -> không cảnh báo
    # - Trên 60%: có thể camera bị che -> không cảnh báo
    is_large = (area_ratio >= OBSTACLE_MIN_RATIO and 
                area_ratio <= OBSTACLE_MAX_RATIO and
                area >= OBSTACLE_MIN_AREA)
    
    # Tính "độ lớn" để hiển thị
    if area_ratio < 0.1:
        size_level = "SMALL"
    elif area_ratio < 0.2:
        size_level = "MEDIUM"
    elif area_ratio < 0.35:
        size_level = "LARGE"
    else:
        size_level = "VERY LARGE"
    
    return is_large, area_ratio, size_level

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
            is_large = obstacle.get('is_large', False)
            size_level = obstacle.get('size_level', 'SMALL')
            area_ratio = obstacle.get('area_ratio', 0)
            
            # Chỉ vẽ vật cản LỚN
            if not is_large:
                continue
            
            # Màu dựa trên độ lớn
            if size_level == "VERY LARGE":
                color = (0, 0, 255)      # Đỏ - rất lớn
                thickness = 3
            elif size_level == "LARGE":
                color = (0, 140, 255)    # Cam - lớn
                thickness = 2
            else:
                color = (0, 255, 255)    # Vàng - vừa
                thickness = 1
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            label_text = f"{size_level}: {area_ratio:.1%}"
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
    cv2.putText(frame, f"● {status_text}", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    face_count = len(face_boxes)
    face_text = f"Faces: {face_count}"
    cv2.putText(frame, face_text, 
                (width - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    if OBSTACLE_DETECTION_ENABLED:
        large_obstacles = len([o for o in obstacle_boxes if o.get('is_large', False)])
        obs_color = (100, 100, 255) if large_obstacles > 0 else (200, 200, 200)
        obs_text = f"Large Obstacles: {large_obstacles}"
        cv2.putText(frame, obs_text, 
                    (width - 130, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obs_color, 1)
    
    time_text = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, time_text, 
                (width - 100, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if face_stability_counter > 0:
        stability_text = f"Face Stable: {face_stability_counter}/{FACE_STABILITY_THRESHOLD}"
        stability_color = (0, 255, 0) if face_stability_counter >= FACE_STABILITY_THRESHOLD else (0, 255, 255)
        cv2.putText(frame, stability_text, 
                    (10, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, stability_color, 1)
    
    priority_status = check_audio_priority()
    priority_color = (0, 255, 0) if priority_status == "FACE_PRIORITY" else (255, 255, 0)
    priority_text = f"Audio Priority: {'FACE' if priority_status == 'FACE_PRIORITY' else 'FREE'}"
    cv2.putText(frame, priority_text, 
                (10, height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, priority_color, 1)
    
    if last_face_result[0] != "UNKNOWN" and time.time() - last_face_detection_time < 5:
        name, conf, _ = last_face_result
        result_text = f"Last: {name} ({conf:.1%})"
        cv2.putText(frame, result_text, 
                    (width - 150, height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return frame

# ================================================================
# HÀM GỬI KẾT QUẢ VỀ PI
# ================================================================
def send_to_pi_async(name, confidence, is_obstacle=False, danger_level=0, obstacle_count=0):
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
            print(f"[AUDIO FILTER] Obstacle cooldown: {OBSTACLE_COOLDOWN - (current_time - announce_history['obstacle']):.1f}s left")
            return False
        
        # Ưu tiên âm thanh mặt
        audio_priority = check_audio_priority()
        if audio_priority == "FACE_PRIORITY":
            time_since_face_audio = current_time - last_face_audio_time
            print(f"[AUDIO PRIORITY] Blocking obstacle - Face priority ({time_since_face_audio:.1f}s ago)")
            return False
        
        # Kiểm tra có vật cản lớn không
        if obstacle_count == 0:
            return False
        
        # Confidence threshold
        if confidence < OBSTACLE_CONFIDENCE_THRESHOLD:
            print(f"[OBSTACLE FILTER] Low confidence: {confidence:.1%} < {OBSTACLE_CONFIDENCE_THRESHOLD:.0%}")
            return False
        
        message = "Cẩn thận! Có vật cản phía trước."
        status = "obstacle"
        announce_history["obstacle"] = current_time
        
        print(f"[PI] OBSTACLE ALERT: {message} (confidence: {confidence:.1%}, count: {obstacle_count})")
        
    else:
        # Face recognition
        if name_str == "UNKNOWN":
            if current_time - announce_history["unknown"] < UNKNOWN_COOLDOWN:
                return False
            
            if confidence < FACE_CONFIDENCE_THRESHOLD:
                print(f"[FACE FILTER] Unknown face rejected: {confidence:.1%} < {FACE_CONFIDENCE_THRESHOLD:.0%}")
                return False
            
            if face_stability_counter < FACE_STABILITY_THRESHOLD:
                print(f"[FACE FILTER] Unknown face not stable: {face_stability_counter} < {FACE_STABILITY_THRESHOLD}")
                return False
                
            message = "Có người lạ"
            status = "unknown"
            announce_history["unknown"] = current_time
            print(f"[PI] UNKNOWN: {message} (conf: {confidence:.1%}, stable: {face_stability_counter})")
            
            update_face_audio_time()
            
        else:
            if name_str in announce_history["known"]:
                last_announced = announce_history["known"][name_str]
                if current_time - last_announced < KNOWN_COOLDOWN:
                    return False
            
            if confidence < 0.70:
                print(f"[FACE FILTER] Known face rejected: {confidence:.1%} < 70%")
                return False
            
            if face_stability_counter < FACE_STABILITY_THRESHOLD:
                print(f"[FACE FILTER] Known face not stable: {face_stability_counter} < {FACE_STABILITY_THRESHOLD}")
                return False
            
            message = f"Xin chào {name_str}"
            status = "known"
            announce_history["known"][name_str] = current_time
            print(f"[PI] KNOWN: {name_str} (conf: {confidence:.1%}, stable: {face_stability_counter})")
            
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
            "audio_priority": "FACE" if not is_obstacle else "OBSTACLE"
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
            
            frame_ready.set()
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
# THREAD RECOGNITION
# ================================================================
def recognition_thread():
    """Thread nhận diện khuôn mặt"""
    global last_face_result, last_face_detection_time, current_people, face_boxes
    
    print("[FACE] Face recognition thread started")
    
    temp_dir = "uploads/rtsp_faces"
    os.makedirs(temp_dir, exist_ok=True)
    
    last_process_time = 0
    process_interval = 1.0
    
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
            
            boxes = face_detector.detect_faces_simple(frame)
            face_boxes = boxes
            
            if len(boxes) == 0:
                global face_stability_counter
                face_stability_counter = max(0, face_stability_counter - 1)
                last_process_time = current_time
                continue
            
            x, y, w, h = boxes[0]
            
            if face_stability_counter < FACE_STABILITY_THRESHOLD:
                print(f"[FACE] Face detected but not stable yet: {face_stability_counter}/{FACE_STABILITY_THRESHOLD}")
                last_process_time = current_time
                continue
            
            print(f"[FACE] Processing STABLE face: {w}x{h} (stable: {face_stability_counter})")
            
            margin = 15
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0 or face_crop.shape[0] < 70 or face_crop.shape[1] < 70:
                continue
            
            timestamp = int(time.time() * 1000)
            face_path = os.path.join(temp_dir, f"face_{timestamp}.jpg")
            
            try:
                cv2.imwrite(face_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                if not os.path.exists(face_path):
                    continue
                
                result = recognize_face_func(face_path)
                
                recognized_name = "UNKNOWN"
                confidence = 0.0
                
                if result and isinstance(result, (list, tuple)) and len(result) >= 2:
                    name, conf = result[0], result[1]
                    
                    if not name or name in ["None", "NO_FACE", "NO_EMBEDDING", "", "unknown"]:
                        recognized_name = "UNKNOWN"
                    else:
                        recognized_name = str(name).strip()
                    
                    try:
                        confidence = float(conf) if conf else 0.0
                    except:
                        confidence = 0.0
                    
                    should_send = False
                    
                    if recognized_name == "UNKNOWN":
                        if confidence >= FACE_CONFIDENCE_THRESHOLD:
                            should_send = True
                            print(f"[RESULT] UNKNOWN (conf: {confidence:.1%}) - SENDING")
                        else:
                            print(f"[RESULT] UNKNOWN (conf: {confidence:.1%}) - BELOW THRESHOLD")
                    else:
                        if confidence >= 0.70:
                            should_send = True
                            print(f"[RESULT] {recognized_name} (conf: {confidence:.1%}) - SENDING")
                        else:
                            print(f"[RESULT] {recognized_name} (conf: {confidence:.1%}) - BELOW THRESHOLD")
                    
                    if should_send:
                        send_to_pi_async(recognized_name, confidence, is_obstacle=False)
                        
                        if recognized_name != "UNKNOWN":
                            current_people[recognized_name] = current_time
                        
                        last_face_result = (recognized_name, confidence, face_crop)
                        last_face_detection_time = current_time
                
                last_process_time = current_time
                
            except Exception as e:
                print(f"[FACE] Error: {e}")
            finally:
                if os.path.exists(face_path):
                    try:
                        os.remove(face_path)
                    except:
                        pass
            
            if current_time % 20 < 0.1:
                cleanup_old_people(60)
            
        except Exception as e:
            print(f"[FACE] Thread error: {e}")
            time.sleep(1)
    
    print("[FACE] Recognition thread stopped")

# ================================================================
# THREAD OBSTACLE DETECTION - ĐƠN GIẢN
# ================================================================
def obstacle_detection_thread():
    """Thread phát hiện vật cản LỚN (gần khung hình)"""
    global obstacle_detections, obstacle_boxes, last_obstacle_detection_time
    
    print("[OBSTACLE] Large obstacle detection thread started")
    
    last_process_time = 0
    process_interval = 2.0
    
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
            
            has_obstacle, obstacles, confidence, danger_level = advanced_detector.detect_obstacles(frame)
            
            if hasattr(confidence, 'item'):
                confidence = float(confidence.item())
            else:
                confidence = float(confidence)
            
            filtered_obstacles = []
            large_obstacles_count = 0
            
            if has_obstacle:
                frame_height, frame_width = frame.shape[:2]
                
                for obstacle in obstacles:
                    if 'bbox' not in obstacle:
                        continue
                    
                    obs_confidence = obstacle.get('confidence', 0.5)
                    if obs_confidence < OBSTACLE_CONFIDENCE_THRESHOLD:
                        continue
                    
                    x, y, w, h = obstacle['bbox']
                    
                    # KIỂM TRA VẬT CÓ LỚN KHÔNG (gần khung hình)
                    is_large, area_ratio, size_level = is_large_obstacle(
                        (x, y, w, h), frame_width, frame_height
                    )
                    
                    obstacle['is_large'] = is_large
                    obstacle['area_ratio'] = area_ratio
                    obstacle['size_level'] = size_level
                    
                    # Thêm vào danh sách để hiển thị
                    filtered_obstacles.append(obstacle)
                    
                    # Đếm vật cản LỚN
                    if is_large:
                        large_obstacles_count += 1
                        print(f"[OBSTACLE] Large obstacle detected: {size_level} ({area_ratio:.1%})")
            
            obstacle_detections = filtered_obstacles
            obstacle_boxes = filtered_obstacles.copy()
            
            if len(filtered_obstacles) > 0:
                print(f"[OBSTACLE] Found {len(filtered_obstacles)} obstacles")
                print(f"    Large obstacles (>={OBSTACLE_MIN_RATIO:.0%} of frame): {large_obstacles_count}")
                
                if large_obstacles_count > 0:
                    audio_priority = check_audio_priority()
                    
                    if audio_priority == "OBSTACLE_ALLOWED":
                        new_danger_level = min(3, large_obstacles_count)
                        
                        send_to_pi_async(
                            "OBSTACLE", 
                            confidence, 
                            is_obstacle=True, 
                            danger_level=new_danger_level,
                            obstacle_count=large_obstacles_count
                        )
                    else:
                        time_since_face_audio = current_time - last_face_audio_time
                        print(f"[AUDIO PRIORITY] Blocking obstacle - "
                              f"Face priority ({AUDIO_PRIORITY_TIMEOUT - time_since_face_audio:.1f}s left)")
            
            last_obstacle_detection_time = current_time
            last_process_time = current_time
            
        except Exception as e:
            print(f"[OBSTACLE] Error: {e}")
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
            if camera_connected and display_frame is not None:
                if current_time - last_display_update > 0.3:
                    boxes = face_detector.detect_faces_simple(display_frame)
                    face_boxes = boxes
                    last_display_update = current_time
            
            if camera_connected and display_frame is not None:
                if face_boxes:
                    display_frame = draw_face_boxes(display_frame, face_boxes)
                
                if obstacle_boxes:
                    display_frame = draw_obstacle_boxes(display_frame, obstacle_boxes)
            
            display_frame = draw_overlay_info(display_frame)
            
            if time.time() - last_fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_fps_time = time.time()
                
                fps_color = (100, 255, 100) if fps >= 15 else (100, 100, 255)
                cv2.putText(display_frame, f"FPS: {fps}", 
                           (display_frame.shape[1] - 100, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
            
            ret, buffer = cv2.imencode('.jpg', display_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 80
            ])
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(1/25)
            
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
    print("RTSP SYSTEM - LARGE OBSTACLE DETECTION")
    print("="*60)
    print(f"Camera: {RTSP_URL}")
    print(f"Face Detection: HIGH PRIORITY")
    print(f"  - Min confidence: {FACE_CONFIDENCE_THRESHOLD:.0%}")
    print(f"  - Stability required: {FACE_STABILITY_THRESHOLD} frames")
    print(f"Obstacle Detection: {'LARGE OBJECTS ONLY' if OBSTACLE_DETECTION_ENABLED else 'DISABLED'}")
    print(f"  - Min size ratio: {OBSTACLE_MIN_RATIO:.0%} of frame")
    print(f"  - Max size ratio: {OBSTACLE_MAX_RATIO:.0%} of frame")
    print(f"  - Min area: {OBSTACLE_MIN_AREA}px")
    print(f"Audio Priority: FACE FIRST")
    print(f"  - Face priority timeout: {AUDIO_PRIORITY_TIMEOUT}s")
    print(f"  - Unknown cooldown: {UNKNOWN_COOLDOWN}s")
    print(f"  - Known cooldown: {KNOWN_COOLDOWN}s")
    print(f"  - Obstacle cooldown: {OBSTACLE_COOLDOWN}s")
    print("="*60)
    
    cam_thread = threading.Thread(target=camera_thread, daemon=True, name="CameraThread")
    cam_thread.start()
    time.sleep(2)
    
    face_thread = threading.Thread(target=recognition_thread, daemon=True, name="RecognitionThread")
    face_thread.start()
    print("Face recognition started (HIGH PRIORITY)")
    
    if OBSTACLE_DETECTION_ENABLED and advanced_detector is not None:
        obstacle_thread = threading.Thread(target=obstacle_detection_thread, daemon=True, name="ObstacleThread")
        obstacle_thread.start()
        print("Obstacle detection started (LARGE OBJECTS ONLY)")
    
    print("All threads started")
    print("System running - Only alerts for LARGE obstacles!")
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
    print("RTSP SYSTEM - LARGE OBSTACLE DETECTION")
    
    start_rtsp_system()
    
    try:
        while True:
            time.sleep(15)
            
            print(f"\nSYSTEM STATUS:")
            print(f"  Camera: {'Connected' if camera_connected else 'Disconnected'}")
            print(f"  Face Tracking: {len(current_people)} people")
            
            if len(current_people) > 0:
                recent = list(current_people.keys())[:3]
                print(f"  Recent Faces: {recent}")
            
            print(f"  Face Boxes: {len(face_boxes)}")
            print(f"  Face Stability: {face_stability_counter}/{FACE_STABILITY_THRESHOLD}")
            
            priority_status = check_audio_priority()
            print(f"  Audio Priority: {'FACE' if priority_status == 'FACE_PRIORITY' else 'FREE'}")
            print(f"  Time since last face audio: {time.time() - last_face_audio_time:.1f}s")
            
            if OBSTACLE_DETECTION_ENABLED:
                large_obstacles = len([o for o in obstacle_boxes if o.get('is_large', False)])
                all_obstacles = len(obstacle_boxes)
                print(f"  All obstacles: {all_obstacles}")
                print(f"  Large obstacles (>={OBSTACLE_MIN_RATIO:.0%}): {large_obstacles}")
            
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        stop_rtsp_system()