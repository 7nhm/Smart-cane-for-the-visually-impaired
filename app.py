import sqlite3
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, Response, g, send_file
import sys
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
import time
import requests
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import cv2
import traceback
import json

# =========================================================
# CONFIG PATH & IMPORT Model/
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "visionmate.db")
print(f"Database path: {DB_PATH}")

# ĐỊNH NGHĨA ĐƯỜNG DẪN GỐC CỦA DỰ ÁN
PROJECT_ROOT = r"D:\KHOA_LUAN_TOT_NGHIEP\VISION-MATE\VISION-MATE"
print(f"Project root: {PROJECT_ROOT}")

MODEL_DIR = os.path.join(BASE_DIR, "Model")
sys.path.append(MODEL_DIR)

# Import các module
try:
    from Model.face_db import insert_face, get_all_faces, delete_face as db_delete_face, ensure_table as ensure_face_table
    print("Import face_db successful")
except Exception as e:
    print(f"Import face_db failed: {e}")

try:
    from Model.face_model import get_embedding_from_image_file, recognize_face_from_frame, load_faces, extract_and_save_face_crop, save_recognition_to_history
    print("Import face_model successful")
except Exception as e:
    print(f"Import face_model failed: {e}")
    # Tạo fallback function nếu import thất bại
    def get_embedding_from_image_file(image_path):
        print(f"FALLBACK: No embedding function for {image_path}")
        return None
    
    def load_faces():
        print("FALLBACK: load_faces not available")
        return []
    
    def extract_and_save_face_crop(frame_path, output_dir="uploads/face_crops"):
        print(f"FALLBACK: Cannot extract face crop for {frame_path}")
        return None

try:
    from Model.user import UserModel
    print("Import UserModel class successful")
except Exception as e:
    print(f"Import UserModel class failed: {e}")

try:
    from Model.device_db import device_model
    print("Import device_db successful")
except Exception as e:
    print(f"Import device_db failed: {e}")

import Model.history_db as history_db
history_db.ensure_history_table()

from Model.alert_db import add_alert, get_recent_alerts, ensure_table as ensure_alert_table
from Model.location_model import location_model

# =========================================================
# IMPORT VÀ KHỞI ĐỘNG RTSP SYSTEM ĐỘC LẬP
# =========================================================
try:
    from rtsp_stream import rtsp_stream, start_rtsp_system, stop_rtsp_system
    print("Import rtsp_stream successful")
    
    # KHỞI ĐỘNG HỆ THỐNG RTSP NGAY KHI IMPORT
    start_rtsp_system()
    print("RTSP System started INDEPENDENTLY")
    
    # Biến toàn cục để kiểm tra trạng thái RTSP
    rtsp_system_active = True
    
except ImportError as e:
    print(f"Import rtsp_stream failed: {e}")
    # Tạo fallback function
    def rtsp_stream():
        """Fallback stream function"""
        print("Using fallback RTSP stream")
        
        # Tạo ảnh test
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "RTSP SYSTEM ERROR", (50, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(test_img, "Check rtsp_stream.py", (30, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        while True:
            # Thêm timestamp động
            timestamp = datetime.now().strftime("%H:%M:%S")
            display_img = test_img.copy()
            cv2.putText(display_img, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode(".jpg", display_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.1)
    
    def start_rtsp_system():
        print("RTSP system not available")
        return False
    
    def stop_rtsp_system():
        print("RTSP system not available")
        return False
    
    rtsp_system_active = False

# =========================================================
# FLASK + SOCKET CONFIG
# =========================================================
app = Flask(__name__)
app.secret_key = "visionmate_secret_key_2025"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ĐƯỜNG DẪN CẬP NHẬT - DÙNG ĐƯỜNG DẪN TUYỆT ĐỐI
UPLOADS_ROOT = os.path.join(PROJECT_ROOT, "uploads")
FACE_CROPS_DIR = os.path.join(UPLOADS_ROOT, "face_crops")

app.config["UPLOAD_FOLDER"] = UPLOADS_ROOT
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
app.config["FACE_CROPS_DIR"] = FACE_CROPS_DIR

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(UPLOADS_ROOT, exist_ok=True)
os.makedirs(FACE_CROPS_DIR, exist_ok=True)

print(f"Uploads folder: {UPLOADS_ROOT}")
print(f"Face crops folder: {FACE_CROPS_DIR}")

# =========================================================
# CONFIG RASPBERRY PI
# =========================================================
PI4_BASE_URL = "http://100.93.7.42:5001"

# =========================================================
# HELPER: per-request UserModel
# =========================================================
def get_user_model():
    if "user_model" not in g:
        g.user_model = UserModel(DB_PATH)
    return g.user_model

@app.teardown_appcontext
def close_user_model(exception=None):
    um = g.pop("user_model", None)
    if um is not None:
        try:
            um.close()
        except Exception:
            pass

# =========================================================
# HÀM TẠO ẢNH MẶC ĐỊNH
# =========================================================
def create_default_image(text=""):
    """Tạo ảnh mặc định khi không tìm thấy ảnh gốc"""
    try:
        # Tạo ảnh 150x150 với nền xám
        img = Image.new('RGB', (150, 150), color=(220, 220, 220))
        d = ImageDraw.Draw(img)
        
        # Thêm text
        text_to_show = "No Image" if not text else text[:15]
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Tính toán vị trí text
        text_bbox = d.textbbox((0, 0), text_to_show, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (150 - text_width) // 2
        y = (150 - text_height) // 2
        
        d.text((x, y), text_to_show, fill=(100, 100, 100), font=font)
        
        # Trả về ảnh
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=90)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        print(f"Create default image error: {e}")
        # Trả về ảnh trắng đơn giản nhất
        img = Image.new('RGB', (150, 150), color=(220, 220, 220))
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

# =========================================================
# HELPER: TÌM FILE TRONG CÁC THƯ MỤC
# =========================================================
def find_image_file(image_name):
    """
    Tìm file ảnh trong các thư mục:
    1. face_crops/
    2. uploads/
    3. Toàn bộ cây thư mục uploads (đệ quy)
    """
    # 1. Kiểm tra trong face_crops
    face_crop_path = os.path.join(FACE_CROPS_DIR, image_name)
    if os.path.exists(face_crop_path):
        return face_crop_path
    
    # 2. Kiểm tra trong uploads root
    uploads_path = os.path.join(UPLOADS_ROOT, image_name)
    if os.path.exists(uploads_path):
        return uploads_path
    
    # 3. Tìm đệ quy trong toàn bộ uploads
    for root, dirs, files in os.walk(UPLOADS_ROOT):
        if image_name in files:
            return os.path.join(root, image_name)
    
    return None

# =========================================================
# ROUTE QUAN TRỌNG: LUÔN TRẢ VỀ ẢNH CHO HISTORY
# =========================================================
@app.route("/history_image/<path:image_name>")
def history_image(image_name):
    """
    Route đặc biệt LUÔN trả về ảnh cho history page
    - Tìm ảnh trong tất cả các thư mục
    - Nếu không có, trả về ảnh mặc định
    """
    try:
        print(f"[HISTORY_IMAGE] Looking for: {image_name}")
        
        # Tìm file ảnh
        file_path = find_image_file(image_name)
        
        if file_path:
            print(f"[HISTORY_IMAGE] Found: {file_path}")
            
            # Xác định thư mục gốc và tên file
            dir_name = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            
            return send_from_directory(dir_name, file_name)
        
        # Nếu không tìm thấy
        print(f"[HISTORY_IMAGE] Not found: {image_name}")
        return create_default_image(image_name[:20])
        
    except Exception as e:
        print(f"[HISTORY_IMAGE] Error: {e}")
        return create_default_image("error")

# =========================================================
# ROUTE PHỤC VỤ ẢNH TỪ UPLOADS
# =========================================================
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    try:
        # Tìm file trong các thư mục
        file_path = find_image_file(filename)
        
        if file_path:
            dir_name = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            return send_from_directory(dir_name, file_name)
        
        print(f"File not found: {filename}")
        return create_default_image(filename[:20])
    except Exception as e:
        print(f"Error serving upload {filename}: {e}")
        return create_default_image("error")

# =========================================================
# API ĐỂ KIỂM TRA ĐƯỜNG DẪN
# =========================================================
@app.route("/api/debug/paths", methods=["GET"])
def api_debug_paths():
    """API debug để kiểm tra đường dẫn"""
    return jsonify({
        "success": True,
        "paths": {
            "project_root": PROJECT_ROOT,
            "base_dir": BASE_DIR,
            "uploads_root": UPLOADS_ROOT,
            "face_crops_dir": FACE_CROPS_DIR,
            "database_path": DB_PATH
        },
        "face_crops_exists": os.path.exists(FACE_CROPS_DIR),
        "uploads_exists": os.path.exists(UPLOADS_ROOT),
        "face_crops_files": os.listdir(FACE_CROPS_DIR) if os.path.exists(FACE_CROPS_DIR) else []
    })

# =========================================================
# HÀM GỬI LOG LÊN WEB (KHÔNG GỬI ÂM THANH)
# =========================================================
def send_recognition_log_to_web(name, confidence, image_path=None):
    """
    CHỈ gửi log lên web, KHÔNG gửi âm thanh đến Pi
    (vì RTSP stream đã gửi âm thanh rồi)
    """
    try:
        print(f"[WEB_LOG] Nhận diện: {name} ({confidence:.1%})")
        
        # Lưu vào history database
        if image_path:
            is_cropped = 1 if "face_crops" in str(image_path) else 0
            history_db.add_history(name, "Camera", confidence, os.path.basename(image_path), is_cropped)
        
        # Gửi socket event để web realtime
        socketio.emit("recognize_event", {
            "name": name,
            "confidence": confidence * 100,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    except Exception as e:
        print(f"[WEB_LOG] Lỗi: {e}")
        return False

# =========================================================
# API RECOGNIZE - CHỈ LƯU LOG, KHÔNG GỬI ÂM THANH
# =========================================================
@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    API nhận diện CHỈ lưu log web, không gửi âm thanh
    (RTSP stream đã xử lý âm thanh)
    """
    print("\n" + "=" * 70)
    print("[API_RECOGNIZE] CHỈ LƯU LOG WEB")
    print("=" * 70)
    
    try:
        # Tạo tên file cho ảnh gốc
        original_filename = f"history_{int(time.time()*1000)}.jpg"
        save_path = os.path.join(UPLOADS_ROOT, original_filename)

        file = request.files.get("file")
        if not file:
            print("[API_RECOGNIZE] No file in request")
            return jsonify({"success": False, "error": "No file"}), 400

        try:
            file.save(save_path)
            print(f"[API_RECOGNIZE] Frame saved: {save_path}")
        except Exception as e:
            print(f"[API_RECOGNIZE] Save file failed: {e}")
            return jsonify({"success": False, "error": f"Save file failed: {e}"}), 500

        # LUÔN CẮT KHUÔN MẶT TRƯỚC KHI NHẬN DIỆN
        print("[API_RECOGNIZE] Đang cắt khuôn mặt từ ảnh...")
        face_crop_filename = None
        try:
            face_crop_filename = extract_and_save_face_crop(save_path)
            if face_crop_filename:
                print(f"[API_RECOGNIZE] Đã cắt và lưu khuôn mặt: {face_crop_filename}")
            else:
                print("[API_RECOGNIZE] Không thể cắt khuôn mặt từ ảnh này")
        except Exception as e:
            print(f"[API_RECOGNIZE] Không thể cắt khuôn mặt: {e}")
            face_crop_filename = None

        # NHẬN DIỆN với try-catch
        name = "UNKNOWN"
        confidence = 0.0
        
        try:
            print("[API_RECOGNIZE] Starting face recognition...")
            result = recognize_face_from_frame(save_path)
            print(f"[API_RECOGNIZE] Raw result: {result}")
            
            if result:
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    name = result[0] if result[0] else "UNKNOWN"
                    confidence = float(result[1]) if len(result) > 1 else 0.0
                    print(f"[API_RECOGNIZE] Parsed: name='{name}', confidence={confidence}")
                else:
                    name = str(result) if result else "UNKNOWN"
                    print(f"[API_RECOGNIZE] Simple result: '{name}'")
            else:
                name = "UNKNOWN"
                confidence = 0.0
                print(f"[API_RECOGNIZE] No result, setting to UNKNOWN")
                
        except Exception as e:
            print(f"[API_RECOGNIZE] Recognition error: {e}")
            name = "UNKNOWN"
            confidence = 0.0

        name = name if name and name != "None" else "UNKNOWN"
        
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
            print(f"[API_RECOGNIZE] Confidence: {confidence:.4f}")
        except Exception as e:
            print(f"[API_RECOGNIZE] Error parsing confidence: {e}")
            confidence = 0.0

        print(f"[API_RECOGNIZE] FINAL: Name: '{name}', Confidence: {confidence:.4f}")

        # CHỈ GỬI LOG LÊN WEB, KHÔNG GỬI ÂM THANH
        # Xác định file nào sẽ lưu vào history
        if face_crop_filename:
            filename_to_save = face_crop_filename
            is_cropped_face = 1
        else:
            filename_to_save = original_filename
            is_cropped_face = 0
        
        # Gửi log lên web (lưu history + socket)
        send_recognition_log_to_web(name, confidence, filename_to_save)

        print("=" * 70)
        print("[API_RECOGNIZE] CHỈ LƯU LOG WEB - HOÀN TẤT")
        print("=" * 70)
        
        return jsonify({
            "success": True, 
            "name": name, 
            "confidence": confidence * 100,
            "has_face_crop": bool(face_crop_filename)
        })
        
    except Exception as e:
        print(f"[API_RECOGNIZE] Global error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =========================================================
# API NHẬN CẢNH BÁO VẬT CẢN TỪ RTSP SYSTEM
# =========================================================
@app.route("/api/obstacle_alert", methods=["POST"])
def api_obstacle_alert():
    """API nhận cảnh báo vật cản từ RTSP system"""
    try:
        data = request.json
        print(f"[OBSTACLE_ALERT] Received: {json.dumps(data, indent=2)}")
        
        # Gửi socket event đến dashboard
        socketio.emit("obstacle_alert", {
            "type": "obstacle",
            "count": data.get("count", 0),
            "confidence": data.get("confidence", 0.0),
            "danger_level": data.get("danger_level", 0),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "obstacles": data.get("obstacles", [])
        })
        
        # Cũng gửi alert_event cho backward compatibility
        socketio.emit("alert_event", {
            "type": "obstacle",
            "message": f"Phát hiện {data.get('count', 0)} vật cản (Độ nguy hiểm: {data.get('danger_level', 0)})",
            "count": data.get("count", 0),
            "danger_level": data.get("danger_level", 0),
            "confidence": data.get("confidence", 0.0)
        })
        
        return jsonify({"success": True, "message": "Obstacle alert processed"})
        
    except Exception as e:
        print(f"[OBSTACLE_ALERT] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# =========================================================
# API KIỂM TRA TRẠNG THÁI RTSP SYSTEM
# =========================================================
@app.route("/api/rtsp_status", methods=["GET"])
def api_rtsp_status():
    """API kiểm tra trạng thái RTSP system"""
    try:
        # Kiểm tra xem RTSP system có đang chạy không
        status = "active" if rtsp_system_active else "inactive"
        
        # Kiểm tra xem có module advanced_obstacle_detector không
        try:
            from rtsp_stream import OBSTACLE_DETECTION_ENABLED
            obstacle_enabled = OBSTACLE_DETECTION_ENABLED
        except:
            obstacle_enabled = False
            
        return jsonify({
            "success": True,
            "rtsp_system": status,
            "obstacle_detection": obstacle_enabled,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =========================================================
# API KHỞI ĐỘNG LẠI RTSP SYSTEM
# =========================================================
@app.route("/api/rtsp_restart", methods=["POST"])
def api_rtsp_restart():
    """API khởi động lại RTSP system"""
    try:
        global rtsp_system_active
        
        print("[RTSP_RESTART] Restarting RTSP system...")
        
        # Dừng hệ thống cũ nếu đang chạy
        if rtsp_system_active:
            stop_rtsp_system()
            time.sleep(2)
        
        # Khởi động lại
        start_rtsp_system()
        rtsp_system_active = True
        
        return jsonify({
            "success": True,
            "message": "RTSP system restarted successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[RTSP_RESTART] Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =========================================================
# TEST ROUTES ĐƠN GIẢN
# =========================================================
@app.route("/api/test_pi_connection", methods=["GET"])
def test_pi_connection():
    """Test kết nối đến Pi"""
    try:
        response = requests.get(f"{PI4_BASE_URL}/", timeout=3)
        return jsonify({
            "success": True,
            "pi_status": "connected",
            "response": response.text
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "pi_status": "disconnected",
            "error": str(e)
        })

# =========================================================
# API KIỂM TRA TRẠNG THÁI THIẾT BỊ (MỚI THÊM)
# =========================================================
@app.route("/api/device/status", methods=["GET"])
def api_device_status():
    """API kiểm tra trạng thái thiết bị"""
    try:
        print("[API_DEVICE_STATUS] Checking device status...")
        
        # Lấy tất cả thiết bị
        devices = device_model.get_all() if device_model else []
        
        if not devices:
            return jsonify({
                "success": True,
                "status": "offline",
                "message": "No devices found",
                "lastHeartbeat": None,
                "deviceCount": 0
            })
        
        # Lấy thiết bị đầu tiên
        device = devices[0]
        
        # Lấy thông tin heartbeat từ database
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Kiểm tra xem có bảng heartbeat không
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='device_heartbeat'")
        has_heartbeat_table = cur.fetchone() is not None
        
        last_heartbeat = None
        
        if has_heartbeat_table:
            try:
                cur.execute("SELECT last_heartbeat FROM device_heartbeat WHERE device_id = ? ORDER BY id DESC LIMIT 1", (device.deviceID,))
                result = cur.fetchone()
                if result:
                    last_heartbeat = result[0]
            except Exception as e:
                print(f"[API_DEVICE_STATUS] Error reading heartbeat: {e}")
        
        conn.close()
        
        # Kiểm tra trạng thái dựa trên heartbeat
        status = "offline"
        if last_heartbeat:
            try:
                last_time = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                time_diff = datetime.now() - last_time
                
                if time_diff.total_seconds() < 30:  # 30 giây timeout
                    status = "online"
            except Exception as e:
                print(f"[API_DEVICE_STATUS] Error parsing heartbeat time: {e}")
        
        print(f"[API_DEVICE_STATUS] Status: {status}, Last heartbeat: {last_heartbeat}")
        
        return jsonify({
            "success": True,
            "status": status,
            "lastHeartbeat": last_heartbeat,
            "device": {
                "id": device.deviceID,
                "name": device.device_name,
                "sim": device.sim_number
            },
            "deviceCount": len(devices),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[API_DEVICE_STATUS] Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "error"
        }), 500

# =========================================================
# API KHỞI ĐỘNG LẠI THIẾT BỊ (MỚI THÊM)
# =========================================================
@app.route("/api/device/restart/<int:device_id>", methods=["POST"])
def api_device_restart(device_id):
    """API khởi động lại thiết bị"""
    try:
        print(f"[API_DEVICE_RESTART] Restarting device {device_id}")
        
        # Lấy thông tin thiết bị
        devices = device_model.get_all() if device_model else []
        device = None
        for d in devices:
            if hasattr(d, 'deviceID') and d.deviceID == device_id:
                device = d
                break
        
        if not device:
            return jsonify({
                "success": False,
                "error": "Device not found"
            }), 404
        
        # Gửi lệnh restart đến Pi
        try:
            # Gửi HTTP request đến Pi để restart
            response = requests.post(f"{PI4_BASE_URL}/api/restart", timeout=5)
            if response.status_code == 200:
                success = True
                message = "Device restart command sent to Pi"
            else:
                success = False
                message = "Failed to send restart command to Pi"
        except Exception as e:
            print(f"[API_DEVICE_RESTART] Error sending to Pi: {e}")
            success = True  # Vẫn trả về success nếu không gửi được command
            message = "Device will restart when reconnected"
        
        # Cập nhật trạng thái trong database
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            
            # Tạo bảng nếu chưa có
            cur.execute('''CREATE TABLE IF NOT EXISTS device_heartbeat (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            device_id INTEGER,
                            last_heartbeat TIMESTAMP,
                            status TEXT)''')
            
            # Cập nhật heartbeat
            cur.execute('''INSERT INTO device_heartbeat (device_id, last_heartbeat, status) 
                          VALUES (?, ?, ?)''',
                       (device_id, datetime.now().isoformat(), "restarting"))
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"[API_DEVICE_RESTART] Error updating database: {e}")
        
        return jsonify({
            "success": success,
            "message": message,
            "device": {
                "id": device_id,
                "name": device.device_name
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[API_DEVICE_RESTART] Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =========================================================
# ensure DB tables exist
# =========================================================
try:
    ensure_face_table()
except Exception as e:
    print("ensure_face_table failed:", e)

try:
    history_db.ensure_history_table()
except Exception as e:
    print("ensure_history_table failed:", e)

try:
    ensure_alert_table()
except Exception as e:
    print("ensure_alert_table failed:", e)

# =========================================================
# TẠO BẢNG DEVICE_HEARTBEAT NẾU CHƯA CÓ
# =========================================================
def ensure_device_heartbeat_table():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS device_heartbeat (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_id INTEGER,
                        last_heartbeat TIMESTAMP,
                        status TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        print("[DB] Device heartbeat table ensured")
    except Exception as e:
        print(f"[DB] Error creating heartbeat table: {e}")

ensure_device_heartbeat_table()

# =========================================================
# API LẤY VỊ TRÍ MỚI NHẤT (CHO DASHBOARD AUTO-UPDATE)
# =========================================================
@app.route("/api/get-latest-location", methods=["GET"])
def api_get_latest_location():
    """API trả về vị trí mới nhất của thiết bị - luôn fetch từ database"""
    try:
        print("[API] /api/get-latest-location called")
        
        # Lấy tất cả các location từ database (mới nhất trước)
        locations = location_model.get_history(limit=1)
        
        if locations and len(locations) > 0:
            latest = locations[0]
            
            location_data = {
                "lat": latest.get("lat") if isinstance(latest, dict) else getattr(latest, "lat", None),
                "lng": latest.get("lng") if isinstance(latest, dict) else getattr(latest, "lng", None),
                "address": latest.get("address") if isinstance(latest, dict) else getattr(latest, "address", ""),
                "timestamp": latest.get("timestamp") if isinstance(latest, dict) else getattr(latest, "timestamp", None)
            }
            
            print(f"[API] Found location in history: lat={location_data['lat']}, lng={location_data['lng']}")
            
            return jsonify({
                "success": True,
                "location": location_data,
                "source": "location_history"
            })
        
        print("[API] No location data found")
        return jsonify({
            "success": False,
            "error": "No location data available",
            "location": None
        })
        
    except Exception as e:
        print(f"[API] Error in get-latest-location: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "location": None
        }), 500

# =========================================================
# API KIỂM TRA DATABASE
# =========================================================
@app.route("/api/debug/database", methods=["GET"])
def api_debug_database():
    """API debug để kiểm tra database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) as count FROM face_history")
        history_count = cur.fetchone()[0]
        
        cur.execute("SELECT * FROM face_history ORDER BY timestamp DESC LIMIT 5")
        recent_history = cur.fetchall()
        
        cur.execute("SELECT COUNT(*) as count FROM faces")
        faces_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) as count FROM device_heartbeat")
        heartbeat_count = cur.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "success": True,
            "database": {
                "path": DB_PATH,
                "exists": os.path.exists(DB_PATH)
            },
            "tables": {
                "face_history": {
                    "count": history_count,
                    "recent": [
                        {
                            "id": row[0],
                            "face_name": row[1],
                            "device": row[2],
                            "confidence": row[3],
                            "image": row[4],
                            "timestamp": row[5],
                            "is_cropped_face": row[6]
                        } for row in recent_history
                    ]
                },
                "faces": {
                    "count": faces_count
                },
                "device_heartbeat": {
                    "count": heartbeat_count
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =========================================================
# UPLOAD API
# =========================================================
@app.route("/api/file/upload", methods=["POST"])
def api_file_upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không có file được gửi"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Filename rỗng"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOADS_ROOT, filename)
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Không lưu được file: {e}"}), 500

    print(f"FILE SAVED: {os.path.abspath(save_path)}")
    return jsonify({"success": True, "filename": filename})

# =========================================================
# AUTH - LOGIN & SIGNUP (GIỮ NGUYÊN)
# =========================================================
@app.route("/")
def root():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("signin"))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        print("=" * 60)
        print("[SIGNIN] PROCESSING LOGIN")
        
        session.clear()
        
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        
        print(f"Email: {email}")
        
        try:
            user_model = get_user_model()
        except Exception as e:
            print(f"Failed to init UserModel: {e}")
            flash("Hệ thống đang bảo trì!", "error")
            return render_template("pages/signin.html")
        
        try:
            user = user_model.get_by_email(email)
            if not user:
                print(f"User not found: {email}")
                flash("Sai email hoặc mật khẩu!", "error")
                return render_template("pages/signin.html")
            
            print(f"Found user: {user.name}")
            
            if user_model.auth(email, password):
                session["user"] = user.userID
                session["profile"] = {
                    "name": user.name,
                    "email": user.email,
                    "phone": user.phone or "",
                    "address": user.address or "",
                }
                
                print(f"LOGIN SUCCESS - User ID: {user.userID}")
                flash(f"Chào {user.name}!", "success")
                return redirect(url_for("dashboard"))
            else:
                print(f"Wrong password for: {email}")
                flash("Sai email hoặc mật khẩu!", "error")
                
        except Exception as e:
            print(f"Login error: {e}")
            flash("Lỗi đăng nhập!", "error")
        
        print("=" * 60)
        return render_template("pages/signin.html")
    
    return render_template("pages/signin.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        print("=" * 60)
        print("[SIGNUP] PROCESSING REGISTRATION")
        
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "").strip()
        email = request.form.get("email", "").strip().lower()
        
        print(f"Name: {name}, Email: {email}")
        
        try:
            user_model = get_user_model()
        except Exception as e:
            print(f"Failed to init UserModel: {e}")
            flash("Hệ thống đang bảo trì!", "error")
            return render_template("pages/signup.html")
        
        try:
            if not name or not email or not password:
                flash("Vui lòng điền đầy đủ thông tin!", "error")
                return render_template("pages/signup.html")
            
            if "@" not in email or "." not in email:
                flash("Email không hợp lệ!", "error")
                return render_template("pages/signup.html")
            
            if len(password) < 6:
                flash("Mật khẩu phải có ít nhất 6 ký tự!", "error")
                return render_template("pages/signup.html")
            
            existing = user_model.get_by_email(email)
            if existing:
                flash("Email này đã được đăng ký!", "error")
                return render_template("pages/signup.html")
            
            user_id = user_model.add(
                name=name,
                password_plain=password,
                email=email,
                phone="",
                address=""
            )
            
            print(f"User created: ID={user_id}")
            flash("Đăng ký thành công! Vui lòng đăng nhập.", "success")
            return redirect(url_for("signin"))
            
        except Exception as e:
            error_msg = str(e)
            print(f"Signup error: {error_msg}")
            
            if "Email đã tồn tại" in error_msg or "Email đã tồn tại!" in error_msg:
                flash("Email này đã được đăng ký!", "error")
            else:
                flash(f"Lỗi: {error_msg}", "error")
            
            return render_template("pages/signup.html")
    
    return render_template("pages/signup.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất!", "info")
    return redirect(url_for("signin"))

# =========================================================
# DEBUG ROUTES - JSON API
# =========================================================
@app.route("/api/debug/users")
def api_debug_users():
    try:
        user_model = get_user_model()
    except Exception as e:
        return jsonify({"error": "User model not initialized", "detail": str(e)}), 500
    
    try:
        users = user_model.get_all_users()
        current_user_id = session.get("user")
        
        return jsonify({
            "success": True,
            "total_users": len(users),
            "current_user_id": current_user_id,
            "users": users
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/debug/session")
def api_debug_session():
    return jsonify({
        "success": True,
        "session": dict(session)
    })

# =========================================================
# DASHBOARD
# =========================================================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("signin"))

    user_profile = session.get("profile", {})
    devices = device_model.get_all() if device_model else []

    try:
        alerts = get_recent_alerts()
    except Exception as e:
        print("get_recent_alerts failed:", e)
        alerts = []

    return render_template("pages/dashboard.html",
                           user=user_profile,
                           devices=devices,
                           alerts=alerts,
                           now=datetime.now())

# =========================================================
# PROFILE UPDATE
# =========================================================
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect(url_for("signin"))

    user_profile = session.get("profile", {})

    if request.method == "POST":
        user_profile = {
            "name": request.form.get("name"),
            "email": request.form.get("email"),
            "phone": request.form.get("phone"),
            "address": request.form.get("address"),
        }
        session["profile"] = user_profile

        try:
            user_model = get_user_model()
            user_id = session.get("user")
            if user_id:
                user_model.update_user(user_id,
                                       name=user_profile["name"],
                                       email=user_profile["email"],
                                       phone=user_profile["phone"],
                                       address=user_profile["address"])
        except Exception as e:
            print(f"update profile failed: {e}")

        return jsonify({"success": True})

    return render_template("pages/profile.html", user=user_profile)

# =========================================================
# MANAGE FACE - ĐÃ SỬA HOÀN TOÀN VÀ THÊM TRY-CATCH
# =========================================================
@app.route("/manage", methods=["GET", "POST"])
def manage():
    if "user" not in session:
        return redirect(url_for("signin"))

    if request.method == "GET":
        try:
            faces = get_all_faces()
            return render_template("pages/manage.html", faces=faces)
        except Exception as e:
            print(f"Error getting faces: {e}")
            return render_template("pages/manage.html", faces=[])

    # POST request - Thêm khuôn mặt mới
    print("=" * 60)
    print("[MANAGE] ADDING NEW FACE")
    
    try:
        name = request.form.get("name", "").strip()
        relationship = request.form.get("relationship", "").strip()
        email = request.form.get("email", "").strip()
        
        if "face_image" not in request.files:
            return jsonify({"success": False, "error": "Không có file ảnh được gửi"})
        
        file = request.files["face_image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Filename rỗng"})
        
        # Validate input
        if not name:
            return jsonify({"success": False, "error": "Vui lòng nhập tên"})
        
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOADS_ROOT, filename)
        
        try:
            # Save the image
            file.save(save_path)
            print(f"Image saved: {save_path}")
        except Exception as e:
            return jsonify({"success": False, "error": f"Không lưu được file: {e}"}), 500
        
        # Get embedding với try-catch
        print("Getting embedding from image...")
        embedding = None
        try:
            embedding = get_embedding_from_image_file(save_path)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            embedding = None
            
        if embedding is None:
            # Xóa file đã lưu
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except:
                    pass
            return jsonify({"success": False, "error": "Không tìm thấy khuôn mặt trong ảnh"})
        
        # Convert embedding
        emb_to_store = None
        try:
            if hasattr(embedding, "tolist"):
                emb_to_store = embedding.tolist()
            elif isinstance(embedding, (list, tuple)):
                emb_to_store = list(embedding)
            else:
                emb_to_store = [float(embedding)]
        except Exception as e:
            print(f"Error converting embedding: {e}")
            # Xóa file đã lưu
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except:
                    pass
            return jsonify({"success": False, "error": "Lỗi xử lý dữ liệu khuôn mặt"})
        
        # Insert into database
        try:
            success = insert_face(
                name=name,
                relationship=relationship,
                email=email,
                image_path=filename,
                embedding=emb_to_store,
            )
            
            if success:
                # Reload faces cache
                try:
                    load_faces()
                    print("Faces cache reloaded after insert")
                except Exception as e:
                    print(f"Cannot reload faces cache: {e}")
                
                return jsonify({
                    "success": True,
                    "message": f"Đã thêm {name} vào hệ thống"
                })
            else:
                # Xóa file đã lưu
                if os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                    except:
                        pass
                return jsonify({
                    "success": False,
                    "error": "Không thể thêm khuôn mặt vào database"
                })
                
        except Exception as e:
            print(f"Insert face error: {e}")
            # Xóa file đã lưu
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except:
                    pass
            return jsonify({"success": False, "error": str(e)})
            
    except Exception as e:
        print(f"General error in manage POST: {e}")
        return jsonify({"success": False, "error": "Lỗi hệ thống"}), 500

@app.route("/manage/delete/<int:id>", methods=["POST"])
def delete_face(id):
    """API xóa khuôn mặt - ĐÃ FIX"""
    try:
        print(f"[MANAGE] Deleting face ID: {id}")
        
        # Gọi hàm xóa từ face_db
        success = db_delete_face(id)
        
        if success:
            # Reload faces cache
            try:
                load_faces()
                print(f"Face {id} deleted successfully")
            except Exception as e:
                print(f"Cannot reload faces cache: {e}")
            
            return jsonify({
                "success": True,
                "message": "Đã xóa khuôn mặt thành công"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Không tìm thấy khuôn mặt để xóa"
            })
            
    except Exception as e:
        print(f"Delete face error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

# =========================================================
# API KIỂM TRA KHUÔN MẶT
# =========================================================
@app.route("/api/check_face", methods=["POST"])
def api_check_face():
    """API kiểm tra xem ảnh có khuôn mặt không"""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không có file"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Filename rỗng"})
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOADS_ROOT, filename)
    
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Không lưu được file: {e}"})
    
    # Check if face exists
    try:
        from deepface import DeepFace
        face_objs = DeepFace.extract_faces(
            img_path=save_path,
            detector_backend="opencv",
            enforce_detection=False
        )
        
        count = len(face_objs) if face_objs else 0
        
        # Clean up temp file
        if os.path.exists(save_path):
            os.remove(save_path)
        
        return jsonify({
            "success": True,
            "has_face": count > 0,
            "count": count
        })
        
    except Exception as e:
        # Clean up temp file
        if os.path.exists(save_path):
            os.remove(save_path)
        
        return jsonify({
            "success": False,
            "error": str(e),
            "has_face": False,
            "count": 0
        })

# =========================================================
# DEVICE PAGE
# =========================================================
@app.route("/device", methods=["GET", "POST"])
def device():
    if "user" not in session:
        return redirect(url_for("signin"))

    if request.method == "POST":
        device_model.add(request.form.get("device_name"),
                         request.form.get("sim_number"))
        flash("Thêm thiết bị thành công!", "success")
        return redirect(url_for("device"))

    return render_template("pages/device.html", devices=device_model.get_all())

@app.route("/device/delete/<int:id>", methods=["POST"])
def delete_device(id):
    device_model.delete(id)
    flash("Đã xoá thiết bị!", "success")
    return redirect(url_for("device"))

# =========================================================
# RTSP CAMERA STREAM - CHỈ ĐỂ HIỂN THỊ, AI ĐÃ CHẠY ĐỘC LẬP
# =========================================================
@app.route("/stream")
def stream():
    """Stream video từ camera RTSP - CHỈ HIỂN THỊ, AI đã chạy độc lập"""
    print(f"[WEB_STREAM] Bắt đầu stream cho web (chỉ hiển thị)")
    return Response(rtsp_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================================================
# API TỪ RASPBERRY PI (SOS, LOCATION, ALERT)
# =========================================================
@app.route("/api/sos", methods=["POST"])
def api_sos():
    try:
        data = request.json or {}
        print("[SERVER] SOS RECEIVED")

        title = data.get("title", "SOS Alert")
        message = data.get("message", "Thiết bị gửi tín hiệu SOS")
        device = data.get("device", "Unknown Device")
        gps = data.get("gps", {})
        
        lat = gps.get("lat")
        lng = gps.get("lng") or gps.get("lon")

        print(f"GPS Data: lat={lat}, lng={lng}, device={device}")

        try:
            add_alert(title, f"{message} | Device: {device}")
        except Exception as e:
            print("add_alert failed:", e)

        if lat and lng:
            try:
                location_model.insert(float(lat), float(lng), device, f"SOS: {message}")
                print("Đã lưu SOS location")
            except Exception as e:
                print("location_model.insert failed:", e)

        try:
            socketio.emit("alert_event", {
                "title": title,
                "message": message,
                "device": device,
                "gps": {"lat": lat, "lng": lng}
            })
            
            if lat and lng:
                socketio.emit("location_event", {
                    "lat": float(lat),
                    "lng": float(lng),
                    "device": device,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            print("Socket emit failed:", e)

        return jsonify({
            "success": True, 
            "message": "SOS received"
        }), 200
        
    except Exception as e:
        print(f"Lỗi xử lý SOS: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/location", methods=["POST"])
def api_location():
    try:
        if request.is_json:
            data = request.json
            device = data.get("device", "Unknown Device")
            lat = data.get("lat")
            lng = data.get("lng") or data.get("lon")
        else:
            device = request.form.get("device", "Unknown Device")
            lat = request.form.get("lat")
            lng = request.form.get("lng") or request.form.get("lon")
        
        if not lat or not lng:
            return jsonify({"success": False, "error": "Missing lat/lng"}), 400
        
        lat_float = float(lat)
        lng_float = float(lng)
        
        location_model.insert(lat_float, lng_float, device)
        
        socketio.emit("location_event", {
            "lat": lat_float,
            "lng": lng_float,
            "device": device,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "success": True, 
            "message": "Location saved"
        })
        
    except Exception as e:
        print(f"Lỗi xử lý location: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/alert", methods=["POST"])
def api_alert():
    title = request.json.get("title")
    message = request.json.get("message")
    try:
        add_alert(title, message)
    except Exception as e:
        print("add_alert failed:", e)
    try:
        socketio.emit("alert_event", {"title": title, "message": message})
    except:
        pass
    return jsonify({"success": True})

# =========================================================
# API LÀM MỚI VỊ TRÍ THIẾT BỊ (ĐÃ SỬA LỖI)
# =========================================================
@app.route("/api/device/refresh_location/<int:device_id>", methods=["POST"])
def api_device_refresh_location(device_id):
    """API làm mới vị trí cho thiết bị cụ thể - ĐÃ SỬA LỖI"""
    try:
        print(f"[DEVICE_REFRESH] Request refresh location for device {device_id}")
        
        # Kiểm tra xem device có tồn tại không
        devices = device_model.get_all() if device_model else []
        
        print(f"Total devices: {len(devices)}")
        for device in devices:
            # In thông tin device để debug
            print(f"Device object: {device}")
            print(f"Device type: {type(device)}")
            
            # Kiểm tra các thuộc tính có thể có
            if hasattr(device, '__dict__'):
                print(f"Device attributes: {device.__dict__}")
        
        device_exists = False
        device_name = f"Device {device_id}"
        
        # CÁCH MỚI: Kiểm tra theo nhiều cách khác nhau
        for device in devices:
            # Cách 1: Kiểm tra thuộc tính deviceID
            if hasattr(device, 'deviceID') and device.deviceID == device_id:
                device_exists = True
                device_name = getattr(device, 'deviceName', f"Device {device_id}")
                print(f"Found device by deviceID: {device_name}")
                break
            
            # Cách 2: Kiểm tra thuộc tính id
            elif hasattr(device, 'id') and device.id == device_id:
                device_exists = True
                device_name = getattr(device, 'deviceName', f"Device {device_id}")
                print(f"Found device by id: {device_name}")
                break
            
            # Cách 3: Kiểm tra dictionary key
            elif isinstance(device, dict) and device.get('deviceID') == device_id:
                device_exists = True
                device_name = device.get('deviceName', f"Device {device_id}")
                print(f"Found device by dict key: {device_name}")
                break
        
        if not device_exists:
            print(f"[DEVICE_REFRESH] Device {device_id} not found. Using default name.")
            # Vẫn tiếp tục xử lý với device name mặc định
        
        # Lấy vị trí mới nhất từ database
        latest_location = location_model.get_latest()
        
        if latest_location:
            # Xử lý latest_location nếu là object
            if hasattr(latest_location, '__dict__'):
                location_dict = latest_location.__dict__
            elif isinstance(latest_location, dict):
                location_dict = latest_location
            else:
                location_dict = {}
            
            lat = location_dict.get('lat') if isinstance(location_dict, dict) else getattr(latest_location, 'lat', 0.0)
            lng = location_dict.get('lng') if isinstance(location_dict, dict) else getattr(latest_location, 'lng', 0.0)
            address = location_dict.get('address') if isinstance(location_dict, dict) else getattr(latest_location, 'address', '')
            
            # Gửi vị trí mới qua socket
            socketio.emit("location_event", {
                "lat": float(lat) if lat else 0.0,
                "lng": float(lng) if lng else 0.0,
                "device": device_name,
                "address": address or "",
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"[DEVICE_REFRESH] Sent location event for {device_name}: lat={lat}, lng={lng}")
            
            return jsonify({
                "success": True,
                "message": f"Location refreshed for device {device_id}",
                "location": {
                    "lat": lat,
                    "lng": lng,
                    "address": address,
                    "device": device_name
                }
            })
        else:
            print("[DEVICE_REFRESH] No location data available")
            return jsonify({
                "success": False,
                "error": "No location data available"
            }), 404
            
    except Exception as e:
        print(f"[DEVICE_REFRESH] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# =========================================================
# API LẤY LOCATIONS VỚI FILTER NGÀY
# =========================================================
@app.route("/api/locations/date_range", methods=["GET"])
def api_locations_date_range():
    """API lấy locations theo khoảng thời gian"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        print(f"[LOCATIONS_DATE_RANGE] Request with start={start_date}, end={end_date}")
        
        # Lấy tất cả locations
        all_locations = location_model.get_history(limit=500)
        
        if not all_locations:
            return jsonify({
                "success": True,
                "locations": [],
                "count": 0,
                "message": "No location data available"
            })
        
        # Lọc theo ngày nếu có
        filtered_locations = []
        
        for loc in all_locations:
            # Xử lý loc có thể là object hoặc dict
            if hasattr(loc, 'timestamp'):
                loc_time = loc.timestamp
            elif isinstance(loc, dict):
                loc_time = loc.get("timestamp", "")
            else:
                loc_time = ""
                
            if not loc_time:
                continue
                
            # Chuyển timestamp thành datetime object
            try:
                if isinstance(loc_time, str):
                    # Parse timestamp string
                    loc_datetime = datetime.fromisoformat(loc_time.replace('Z', '+00:00'))
                else:
                    # Giả sử là datetime object
                    loc_datetime = loc_time
                
                # Lọc theo start_date
                if start_date:
                    start_datetime = datetime.fromisoformat(start_date + "T00:00:00")
                    if loc_datetime.date() < start_datetime.date():
                        continue
                
                # Lọc theo end_date
                if end_date:
                    end_datetime = datetime.fromisoformat(end_date + "T23:59:59")
                    if loc_datetime.date() > end_datetime.date():
                        continue
                
                # Thêm location vào kết quả
                if isinstance(loc, dict):
                    filtered_locations.append(loc)
                else:
                    # Chuyển object thành dict
                    loc_dict = {
                        "lat": getattr(loc, 'lat', 0.0),
                        "lng": getattr(loc, 'lng', 0.0),
                        "device": getattr(loc, 'device', "Unknown Device"),
                        "address": getattr(loc, 'address', ""),
                        "timestamp": loc_time
                    }
                    filtered_locations.append(loc_dict)
                
            except Exception as e:
                print(f"Error parsing date for location: {e}")
                continue
        
        return jsonify({
            "success": True, 
            "locations": filtered_locations,
            "count": len(filtered_locations)
        })
        
    except Exception as e:
        print(f"Lỗi khi lấy locations với filter ngày: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "locations": []}), 500

# =========================================================
# API TO GET ALL LOCATIONS (ĐÃ SỬA)
# =========================================================
@app.route("/api/locations", methods=["GET"])
def api_all_locations():
    try:
        # Kiểm tra xem có filter ngày không
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date or end_date:
            # Gọi API filter ngày
            return api_locations_date_range()
        
        # Nếu không có filter, trả về tất cả
        history = location_model.get_history(limit=50)
        
        print(f"API /api/locations trả về {len(history)} locations")
        
        formatted_locations = []
        for loc in history:
            # Xử lý cả object và dict
            if isinstance(loc, dict):
                formatted_locations.append({
                    "lat": loc.get("lat", 0.0),
                    "lng": loc.get("lng", 0.0),
                    "device": loc.get("device", "Unknown Device"),
                    "address": loc.get("address", ""),
                    "timestamp": loc.get("timestamp", datetime.now().isoformat())
                })
            else:
                # Nếu là object
                formatted_locations.append({
                    "lat": getattr(loc, 'lat', 0.0),
                    "lng": getattr(loc, 'lng', 0.0),
                    "device": getattr(loc, 'device', "Unknown Device"),
                    "address": getattr(loc, 'address', ""),
                    "timestamp": getattr(loc, 'timestamp', datetime.now().isoformat())
                })
        
        return jsonify({
            "success": True, 
            "locations": formatted_locations,
            "count": len(formatted_locations)
        })
        
    except Exception as e:
        print(f"Lỗi khi lấy locations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e), "locations": []}), 500

# =========================================================
# API CLEAR LOCATION HISTORY
# =========================================================
@app.route("/api/locations/clear", methods=["POST"])
def api_clear_locations():
    try:
        count = location_model.clear_history()
        return jsonify({
            "success": True, 
            "message": f"Đã xóa {count} bản ghi",
            "count": count
        })
    except Exception as e:
        print(f"Lỗi khi xóa locations: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# =========================================================
# OTHER PAGES
# =========================================================
@app.route("/history")
def history():
    try:
        history_data = history_db.get_all_history()
        print(f"History count: {len(history_data)}")
        
        # Format lại dữ liệu để đảm bảo có ảnh
        formatted_history = []
        for item in history_data:
            face_name = getattr(item, 'face_name', 'UNKNOWN')
            if not face_name or face_name == "None":
                face_name = "UNKNOWN"
            
            image_url = f"/history_image/{getattr(item, 'image', '')}"
            
            formatted_history.append({
                "id": getattr(item, 'id', 0),
                "face_name": face_name,
                "device": getattr(item, 'device', 'Camera'),
                "confidence": float(getattr(item, 'confidence', 0.0)) * 100,
                "image": getattr(item, 'image', ''),
                "image_url": image_url,
                "timestamp": getattr(item, 'timestamp', ''),
                "is_cropped_face": getattr(item, 'is_cropped_face', 0)
            })
        
        return render_template("pages/history.html", history=formatted_history)
    except Exception as e:
        print(f"Error loading history: {e}")
        return render_template("pages/history.html", history=[])

@app.route("/location")
def location():
    try:
        latest = location_model.get_latest()
        history = location_model.get_history(20)
        
        # Format dữ liệu
        formatted_latest = None
        if latest:
            if isinstance(latest, dict):
                formatted_latest = latest
            else:
                formatted_latest = {
                    "lat": getattr(latest, 'lat', 0.0),
                    "lng": getattr(latest, 'lng', 0.0),
                    "device": getattr(latest, 'device', "Unknown Device"),
                    "address": getattr(latest, 'address', ""),
                    "timestamp": getattr(latest, 'timestamp', "")
                }
        
        formatted_history = []
        for loc in history:
            if isinstance(loc, dict):
                formatted_history.append(loc)
            else:
                formatted_history.append({
                    "lat": getattr(loc, 'lat', 0.0),
                    "lng": getattr(loc, 'lng', 0.0),
                    "device": getattr(loc, 'device', "Unknown Device"),
                    "address": getattr(loc, 'address', ""),
                    "timestamp": getattr(loc, 'timestamp', "")
                })
        
        return render_template("pages/location.html",
                               latest=formatted_latest,
                               history=formatted_history)
    except Exception as e:
        print(f"Lỗi khi load trang location: {e}")
        import traceback
        traceback.print_exc()
        return render_template("pages/location.html",
                               latest=None,
                               history=[])

@app.route("/warning")
def warning():
    try:
        alerts = get_recent_alerts(10)
    except Exception as e:
        print("get_recent_alerts failed:", e)
        alerts = []
    return render_template("pages/warning.html", alerts=alerts)

# =========================================================
# API: HEARTBEAT DEVICE
# =========================================================
@app.route("/api/device/heartbeat", methods=["POST"])
def api_device_heartbeat():
    data = request.json or {}
    device_name = data.get("device")
    if not device_name:
        return jsonify({"success": False, "error": "Missing device name"}), 400

    try:
        # Cập nhật heartbeat trong database
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Tìm device ID
        cur.execute("SELECT deviceID FROM devices WHERE device_name = ?", (device_name,))
        device_result = cur.fetchone()
        
        if device_result:
            device_id = device_result[0]
            
            # Cập nhật heartbeat
            cur.execute('''INSERT INTO device_heartbeat (device_id, last_heartbeat, status) 
                          VALUES (?, ?, ?)''',
                       (device_id, datetime.now().isoformat(), "online"))
            conn.commit()
            
            print(f"[HEARTBEAT] Device {device_name} (ID: {device_id}) heartbeat updated")
        
        conn.close()
        
    except Exception as e:
        print("Heartbeat update failed:", e)
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": True, "message": f"{device_name} heartbeat updated"})

# =========================================================
# API LẤY HISTORY VỚI ẢNH ĐẦY ĐỦ
# =========================================================
@app.route("/api/history/full", methods=["GET"])
def api_history_full():
    """Lấy toàn bộ lịch sử nhận diện với URL ảnh CHÍNH XÁC"""
    try:
        history_data = history_db.get_all_history()
        
        formatted_history = []
        for item in history_data:
            image_url = f"/history_image/{getattr(item, 'image', '')}"
            
            face_name = getattr(item, 'face_name', 'UNKNOWN')
            if not face_name or face_name == "None":
                face_name = "UNKNOWN"
            
            formatted_history.append({
                "id": getattr(item, 'id', 0),
                "face_name": face_name,
                "device": getattr(item, 'device', 'Camera'),
                "confidence": float(getattr(item, 'confidence', 0.0)) * 100,
                "confidence_raw": float(getattr(item, 'confidence', 0.0)),
                "image": getattr(item, 'image', ''),
                "image_url": image_url,
                "timestamp": getattr(item, 'timestamp', ''),
                "is_cropped_face": getattr(item, 'is_cropped_face', 0),
                "has_image": True
            })
        
        print(f"[API_HISTORY] Returned {len(formatted_history)} items with images")
        return jsonify({
            "success": True,
            "count": len(formatted_history),
            "history": formatted_history
        })
        
    except Exception as e:
        print(f"Lỗi khi lấy history: {e}")
        return jsonify({"success": False, "error": str(e), "history": []}), 500

# =========================================================
# API XÓA LỊCH SỬ
# =========================================================
@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Xóa toàn bộ lịch sử"""
    try:
        count = history_db.clear_all_history()
        return jsonify({
            "success": True,
            "message": f"Đã xóa {count} bản ghi lịch sử",
            "count": count
        })
    except Exception as e:
        print(f"Lỗi khi xóa history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# =========================================================
# API THÊM KHUÔN MẶT TỪ HISTORY (JSON API)
# =========================================================
@app.route("/api/add_face_from_history", methods=["POST"])
def api_add_face_from_history():
    """Thêm khuôn mặt mới từ ảnh trong history (JSON API)"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Chưa đăng nhập"}), 401
    
    if not request.is_json:
        return jsonify({"success": False, "error": "Yêu cầu phải là JSON"}), 400
    
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "Thiếu dữ liệu"}), 400
    
    name = data.get("name", "").strip()
    image_path = data.get("image_path", "").strip()
    relationship = data.get("relationship", "")
    email = data.get("email", "")
    
    if not name or not image_path:
        return jsonify({"success": False, "error": "Thiếu tên hoặc ảnh"}), 400
    
    try:
        # Tìm file ảnh
        full_image_path = find_image_file(image_path)
        if not full_image_path:
            return jsonify({"success": False, "error": f"Ảnh không tồn tại: {image_path}"}), 400
        
        print(f"Adding face from JSON API: {full_image_path}")
        
        # Lấy embedding từ ảnh
        embedding = get_embedding_from_image_file(full_image_path)
        if embedding is None:
            return jsonify({"success": False, "error": "Không tìm thấy khuôn mặt trong ảnh"}), 400
        
        # Chuyển embedding về list
        if hasattr(embedding, "tolist"):
            emb_to_store = embedding.tolist()
        else:
            emb_to_store = list(embedding)
        
        # Lưu vào database
        filename = os.path.basename(image_path)
        success = insert_face(
            name=name,
            relationship=relationship,
            email=email,
            image_path=filename,
            embedding=emb_to_store,
        )
        
        if success:
            # Reload cache
            try:
                load_faces()
            except Exception as e:
                print(f"Cannot reload faces cache: {e}")
            
            print(f"Đã thêm khuôn mặt mới từ JSON API: {name}")
            
            return jsonify({
                "success": True,
                "message": f"Đã thêm {name} vào hệ thống"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Không thể lưu vào database"
            })
        
    except Exception as e:
        print(f"Lỗi khi thêm khuôn mặt từ JSON API: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# =========================================================
# TẠO THƯ MỤC STATIC VÀ ẢNH MẶC ĐỊNH
# =========================================================
def create_static_dir():
    """Tạo thư mục static và ảnh mặc định nếu chưa có"""
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    
    default_image_path = os.path.join(static_dir, "default_face.jpg")
    if not os.path.exists(default_image_path):
        try:
            img = Image.new('RGB', (150, 150), color=(220, 220, 220))
            d = ImageDraw.Draw(img)
            d.text((50, 65), "No Image", fill=(150, 150, 150))
            img.save(default_image_path)
            print(f"Created default image: {default_image_path}")
        except Exception as e:
            print(f"Cannot create default image: {e}")

create_static_dir()

# =========================================================
# CLEANUP ON EXIT
# =========================================================
import atexit

@atexit.register
def cleanup_on_exit():
    """Dọn dẹp khi server dừng"""
    print("\n" + "=" * 60)
    print("CLEANUP ON EXIT")
    print("=" * 60)
    
    try:
        if rtsp_system_active:
            print("Stopping RTSP system...")
            stop_rtsp_system()
            print("RTSP system stopped")
    except Exception as e:
        print(f"Error stopping RTSP system: {e}")
    
    print("Cleanup completed")

# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VISION-MATE STARTED (INDEPENDENT RTSP SYSTEM)")
    print(f"URL: http://localhost:5000")
    print(f"Database: {DB_PATH}")
    print(f"RTSP URL: rtsp://100.93.7.42:8554/usb")
    print("\nTÍNH NĂNG:")
    print("• RTSP System: CHẠY ĐỘC LẬP từ khi server start")
    print("• Face + Obstacle Detection: Hoạt động 24/7")
    print("• Audio Announcement: Trực tiếp trên Pi")
    print("• Web Dashboard: Hiển thị real-time + quản lý")
    print("• Obstacle Alerts: Hiển thị trên dashboard")
    print("• Location Tracking: Real-time map với filter")
    print("• Device Status Monitoring: Real-time device status")
    print("=" * 60)
    
    try:
        socketio.run(app, 
                    host="0.0.0.0", 
                    port=5000, 
                    debug=True, 
                    use_reloader=True,
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    finally:
        cleanup_on_exit()