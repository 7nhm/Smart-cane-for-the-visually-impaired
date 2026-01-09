import os
import sqlite3
import numpy as np
import json
import cv2
import time
import threading
from deepface import DeepFace
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import warnings
import datetime
import sys
# thiết lập biến toàn cục và đường dẫn
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # thư mục gốc
PROJECT_ROOT = os.path.dirname(BASE_DIR) # thư mục dự án
DB_PATH = os.path.join(PROJECT_ROOT, "visionmate.db")
UPLOADS_ROOT = os.path.join(PROJECT_ROOT, "uploads")
FACE_CROPS_DIR = os.path.join(UPLOADS_ROOT, "face_crops")

print(f"[FACE_MODEL] Project root: {PROJECT_ROOT}")
print(f"[FACE_MODEL] Face crops dir: {FACE_CROPS_DIR}")

# Đảm bảo thư mục tồn tại
os.makedirs(FACE_CROPS_DIR, exist_ok=True)

print("Loading DeepFace model (Facenet512)...")
MODEL_NAME = "Facenet512"
try:
    model = DeepFace.build_model(MODEL_NAME)
    print("DeepFace model ready!")
except Exception as e:
    print(f"Error loading DeepFace model: {e}")
    model = None

# ========================================
# IMPORT HISTORY_DB ĐỂ LƯU LỊCH SỬ
# ========================================
def init_history_connection():
    """Kết nối với history_db"""
    try:
        # Import history_db từ cùng project
        project_root = os.path.dirname(BASE_DIR)
        sys.path.append(project_root)
        
        from history_db import add_history, ensure_history_table
        
        # Đảm bảo bảng tồn tại
        ensure_history_table()
        print("History DB initialized")
        
        return add_history
    except Exception as e:
        print(f"Cannot import history_db: {e}")
        # Tạo fallback function
        def fallback_add_history(face_name, device, confidence, image, is_cropped_face=0):
            print(f"[FALLBACK HISTORY] {face_name} ({confidence:.2%}) - {image}")
            return 0
        return fallback_add_history

# Khởi tạo hàm lưu history
save_history_func = init_history_connection()

# ========================================
# RAM CACHE: embedding cache với version control
# ========================================
faces_cache = []
cache_lock = threading.Lock()
cache_version = 1
last_cache_update = time.time()
cache_sync_interval = 10

def get_faces_from_db():
    """Lấy faces từ database"""
    try:
        if not os.path.exists(DB_PATH):
            print("DB not found")
            return []
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
        if not cursor.fetchone():
            print("Table 'faces' does not exist")
            conn.close()
            return []
        
        cursor.execute("SELECT faceID, name, relationship, email, image_path, embedding FROM faces")
        rows = cursor.fetchall()
        conn.close()
        
        faces = []
        for r in rows:
            try:
                embedding_data = json.loads(r[5])
                if len(embedding_data) >= 128:
                    faces.append({
                        "id": r[0],
                        "name": r[1] or "Unknown",
                        "relationship": r[2] or "Unknown",
                        "email": r[3] or "",
                        "image_path": r[4] or "",
                        "embedding": np.array(embedding_data, dtype=np.float32)
                    })
                else:
                    print(f"Invalid embedding length for {r[1]}: {len(embedding_data)}")
            except Exception as e:
                print(f"Error parsing face {r[0]}: {e}")
                continue
        
        return faces
        
    except Exception as e:
        print(f"Error loading from DB: {e}")
        traceback.print_exc()
        return []

def sync_faces_cache():
    """Đồng bộ cache từ DB"""
    global faces_cache, cache_version, last_cache_update
    
    print("[CACHE] Syncing faces cache...")
    
    with cache_lock:
        new_cache = get_faces_from_db()
        faces_cache = new_cache
        cache_version += 1
        last_cache_update = time.time()
        
        print(f"[CACHE] Synced: {len(faces_cache)} faces (v{cache_version})")
        
        if faces_cache:
            names = [f['name'] for f in faces_cache]
            print(f"Faces: {', '.join(names)}")
        else:
            print("No faces in cache")
        
        return faces_cache

def get_cache_info():
    """Lấy thông tin cache"""
    with cache_lock:
        return {
            "version": cache_version,
            "count": len(faces_cache),
            "last_update": last_cache_update,
            "names": [f["name"] for f in faces_cache[:10]],
            "total_faces": len(faces_cache)
        }

def check_and_sync_cache_if_needed():
    """Tự động sync cache nếu quá cũ"""
    global last_cache_update
    
    current_time = time.time()
    if current_time - last_cache_update > cache_sync_interval:
        print(f"[CACHE] Auto-syncing")
        return sync_faces_cache()
    
    return faces_cache

# ========================================
# HÀM NHẬN DIỆN VÀ LƯU LỊCH SỬ VÀO DB
# ========================================
def recognize_face_from_frame(frame_path, auto_save=True, min_confidence=0.70):
    """Nhận diện khuôn mặt và lưu lịch sử vào DB"""
    
    print(f"\n{'='*60}")
    print(f"[RECOGNITION] Processing: {os.path.basename(frame_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(frame_path):
        print("Frame not found")
        return ("UNKNOWN", 0.0, "Unknown", None)
    
    # Sync cache
    current_cache = check_and_sync_cache_if_needed()
    cache_info = get_cache_info()
    print(f"Cache v{cache_info['version']} - {len(current_cache)} faces")
    
    # Cắt khuôn mặt - LUÔN THỰC HIỆN VÀ KIỂM TRA
    face_crop_filename = None
    try:
        face_crop_filename = extract_and_save_face_crop(frame_path)
        if face_crop_filename:
            # KIỂM TRA FILE CÓ THẬT SỰ TỒN TẠI
            crop_path = os.path.join(FACE_CROPS_DIR, face_crop_filename)
            if os.path.exists(crop_path):
                file_size = os.path.getsize(crop_path)
                print(f"Face crop saved: {face_crop_filename} ({file_size} bytes)")
                
                # Kiểm tra có đọc được không
                test_img = cv2.imread(crop_path)
                if test_img is not None:
                    print(f"File readable, size: {test_img.shape}")
                else:
                    print(f"File saved but cannot be read!")
                    face_crop_filename = None
            else:
                print(f"Face crop file doesn't exist: {crop_path}")
                face_crop_filename = None
        else:
            print("Không thể cắt khuôn mặt từ ảnh này")
    except Exception as e:
        print(f"Cannot crop face: {e}")
        face_crop_filename = None

    # Lấy embedding
    frame_emb = None
    try:
        detectors = ["opencv", "retinaface", "ssd", "mtcnn"]
        
        for detector in detectors:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        DeepFace.represent,
                        img_path=frame_path,
                        model_name=MODEL_NAME,
                        detector_backend=detector,
                        enforce_detection=False,
                        align=True
                    )
                    
                    result = future.result(timeout=20)
                    
                    if result and len(result) > 0:
                        frame_emb = np.array(result[0]["embedding"], dtype=np.float32)
                        print(f"Detected with {detector}")
                        break
            except:
                continue
                
    except Exception as e:
        print(f"DeepFace error: {e}")
        traceback.print_exc()
    
    if frame_emb is None:
        print("Cannot get embedding")
        # LƯU UNKNOWN VÀO DATABASE - LUÔN DÙNG ẢNH GỐC
        if auto_save:
            try:
                history_id = save_history_func(
                    face_name="UNKNOWN",
                    device="RTSP Camera",
                    confidence=0.0,
                    image=os.path.basename(frame_path),
                    is_cropped_face=0  # Luôn là 0 vì không có face crop
                )
                print(f"Saved UNKNOWN to DB (ID: {history_id}) - Original image")
            except Exception as e:
                print(f"Cannot save history: {e}")
        return ("UNKNOWN", 0.0, "Unknown", None)
    
    # So sánh với cache
    print(f"Comparing with {len(current_cache)} faces...")
    
    best_match = None
    best_similarity = 0.0
    best_name = "UNKNOWN"
    
    if len(current_cache) > 0:
        comparisons = []
        
        for idx, face in enumerate(current_cache):
            try:
                similarity = cosine_similarity(frame_emb, face["embedding"])
                comparisons.append((similarity, face))
                
                if idx < 3:
                    print(f"{face['name'][:15]:15} : {similarity:.4f}")
                    
            except Exception as e:
                continue
        
        if comparisons:
            comparisons.sort(key=lambda x: x[0], reverse=True)
            best_similarity, best_match = comparisons[0]
            best_name = best_match["name"]
            
            # Dynamic threshold
            dynamic_threshold = min_confidence
            if len(current_cache) < 5:
                dynamic_threshold = 0.70
            elif len(current_cache) < 10:
                dynamic_threshold = 0.70
            
            print(f"\nBEST MATCH: {best_name} ({best_similarity:.4f})")
            print(f"THRESHOLD: {dynamic_threshold:.4f}")
            
            if best_similarity >= dynamic_threshold:
                confidence_percent = round(best_similarity * 100, 2)
                print(f"MATCH FOUND: {best_name} ({confidence_percent}%)")
                
                # LƯU KNOWN VÀO DATABASE
                if auto_save:
                    try:
                        # QUYẾT ĐỊNH DÙNG ẢNH NÀO
                        if face_crop_filename and os.path.exists(os.path.join(FACE_CROPS_DIR, face_crop_filename)):
                            crop_image = face_crop_filename
                            is_cropped = 1
                            print(f"📸 Using face crop: {crop_image}")
                        else:
                            crop_image = os.path.basename(frame_path)
                            is_cropped = 0
                            print(f"📸 Using original image: {crop_image}")
                        
                        history_id = save_history_func(
                            face_name=best_name,
                            device="RTSP Camera",
                            confidence=best_similarity,
                            image=crop_image,
                            is_cropped_face=is_cropped
                        )
                        print(f"Saved {best_name} to DB (ID: {history_id})")
                    except Exception as e:
                        print(f"Cannot save history: {e}")
                
                return (
                    best_match["name"],
                    best_similarity,
                    best_match.get("relationship", "Unknown"),
                    best_match.get("image_path", None),
                )
    
    print(f"\nNO MATCH FOUND")
    print(f"   Best similarity: {best_similarity:.4f}")
    
    # LƯU UNKNOWN (NO MATCH) VÀO DATABASE
    if auto_save:
        try:
            # QUYẾT ĐỊNH DÙNG ẢNH NÀO
            if face_crop_filename and os.path.exists(os.path.join(FACE_CROPS_DIR, face_crop_filename)):
                crop_image = face_crop_filename
                is_cropped = 1
                print(f"Using face crop: {crop_image}")
            else:
                crop_image = os.path.basename(frame_path)
                is_cropped = 0
                print(f"Using original image: {crop_image}")
            
            history_id = save_history_func(
                face_name="UNKNOWN",
                device="RTSP Camera",
                confidence=best_similarity,
                image=crop_image,
                is_cropped_face=is_cropped
            )
            print(f"Saved UNKNOWN (no match) to DB (ID: {history_id})")
        except Exception as e:
            print(f"Cannot save history: {e}")
    
    return (
        "UNKNOWN",
        best_similarity,
        "Unknown",
        None
    )

# ========================================
#LẤY EMBEDDING TỪ ẢNH 
# ========================================
def get_embedding_from_image_file(image_path):
    """Lấy embedding từ file ảnh"""
    try:
        print(f"Getting embedding from: {os.path.basename(image_path)}")
        
        # Thử các detector khác nhau
        detectors = ["opencv", "retinaface", "ssd", "mtcnn"]
        
        for detector in detectors:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        DeepFace.represent,
                        img_path=image_path,
                        model_name=MODEL_NAME,
                        detector_backend=detector,
                        enforce_detection=False,
                        align=True
                    )
                    
                    result = future.result(timeout=20)
                    
                    if result and len(result) > 0:
                        embedding = np.array(result[0]["embedding"], dtype=np.float32)
                        print(f"Embedding extracted with {detector}: {len(embedding)} dimensions")
                        return embedding
            except Exception as e:
                print(f"Detector {detector} failed: {e}")
                continue
        
        print("No face detected in image")
        return None
        
    except Exception as e:
        print(f"Error getting embedding: {e}")
        traceback.print_exc()
        return None

# ========================================
# CÁC HÀM HỖ TRỢ
# ========================================
def cosine_similarity(a, b):
    """Tính cosine similarity"""
    try:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
            
        similarity = np.dot(a, b) / (a_norm * b_norm)
        similarity = max(-1.0, min(1.0, similarity))
        
        return float(similarity)
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return 0.0

def extract_and_save_face_crop(frame_path, output_dir="uploads/face_crops"):
    """Cắt và lưu khuôn mặt - VERSION FIXED"""
    try:
        # Tạo thư mục với đường dẫn tuyệt đối
        output_dir_abs = os.path.join(PROJECT_ROOT, output_dir)
        os.makedirs(output_dir_abs, exist_ok=True)
        
        print(f"[FACE_CROP] Input: {frame_path}")
        print(f"[FACE_CROP] Output dir: {output_dir_abs}")
        
        if not os.path.exists(frame_path):
            print(f"[FACE_CROP] Image not found: {frame_path}")
            return None
        
        # Đọc ảnh
        img = cv2.imread(frame_path)
        if img is None:
            print(f"[FACE_CROP] Cannot read image: {frame_path}")
            return None
        
        print(f"[FACE_CROP] Image size: {img.shape}")
        
        # Tạo temp file
        temp_dir = os.path.join(PROJECT_ROOT, "uploads", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_{int(time.time()*1000)}.jpg")
        
        try:
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[FACE_CROP] Temp saved: {temp_path}")
        except Exception as e:
            print(f"[FACE_CROP] Cannot save temp: {e}")
            return None
        
        # Detect faces với try-catch
        face_objs = []
        try:
            for detector in ["opencv", "retinaface", "ssd", "mtcnn"]:
                try:
                    face_objs = DeepFace.extract_faces(
                        img_path=temp_path,
                        detector_backend=detector,
                        enforce_detection=False,
                        align=True
                    )
                    if face_objs and len(face_objs) > 0:
                        print(f"[FACE_CROP] Found {len(face_objs)} faces with {detector}")
                        break
                except Exception as e:
                    print(f"Detector {detector} failed: {e}")
                    continue
        finally:
            # Xóa temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        if not face_objs:
            print(f"[FACE_CROP] No face found")
            return None
        
        # Lấy face tốt nhất
        best_face = max(face_objs, key=lambda x: x.get("confidence", 0))
        face_img = best_face["face"]
        
        print(f"[FACE_CROP] Best confidence: {best_face.get('confidence', 0):.3f}")
        print(f"[FACE_CROP] Face shape: {face_img.shape}")
        
        # Convert và lưu
        if face_img.dtype != np.uint8:
            if face_img.max() <= 1.0:
                face_img = (face_img * 255).astype(np.uint8)
        
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        else:
            face_img_bgr = face_img
        
        # Tạo tên file
        timestamp = int(time.time() * 1000)
        face_filename = f"face_crop_{timestamp}.jpg"
        face_path = os.path.join(output_dir_abs, face_filename)
        
        # Lưu file
        success = cv2.imwrite(face_path, face_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            # Kiểm tra lại file
            if os.path.exists(face_path):
                file_size = os.path.getsize(face_path)
                print(f"[FACE_CROP] Face crop saved: {face_path}")
                print(f"[FACE_CROP] File size: {file_size} bytes")
                return face_filename
            else:
                print(f"[FACE_CROP] File was saved but doesn't exist!")
                return None
        else:
            print(f"[FACE_CROP] Failed to save face crop")
            return None
            
    except Exception as e:
        print(f"[FACE_CROP] Error: {e}")
        traceback.print_exc()
        return None

def recognize_face_simple(frame_path):
    """Nhận diện đơn giản cho RTSP stream - CÓ LƯU DATABASE"""
    try:
        check_and_sync_cache_if_needed()
        return recognize_face_from_frame(frame_path, auto_save=True, min_confidence=0.60)
    except Exception as e:
        print(f"Simple recognition error: {e}")
        # Vẫn cố gắng lưu lịch sử lỗi
        try:
            save_history_func(
                face_name="ERROR",
                device="RTSP Camera",
                confidence=0.0,
                image=os.path.basename(frame_path) if frame_path else "unknown",
                is_cropped_face=0
            )
        except:
            pass
        return ("UNKNOWN", 0.0, "Unknown", None)

# Thêm hàm load_faces (nếu chưa có)
def load_faces():
    """Hàm load faces (đồng bộ với sync_faces_cache)"""
    return sync_faces_cache()

# Thêm hàm save_recognition_to_history (fallback)
def save_recognition_to_history(name, confidence, image_path=None):
    """Hàm fallback lưu recognition vào history"""
    try:
        if save_history_func:
            filename = os.path.basename(image_path) if image_path else "unknown"
            save_history_func(
                face_name=name,
                device="RTSP Camera",
                confidence=confidence,
                image=filename,
                is_cropped_face=0
            )
            return True
    except:
        pass
    return False

# Khởi tạo cache
print("Initializing face cache...")
sync_faces_cache()

print(f"Face model module ready! Cache v{cache_version} with {len(faces_cache)} faces")
print(f"Database saving: ENABLED")
print(f"Face crops directory: {FACE_CROPS_DIR}")
print(f"Face crops exists: {os.path.exists(FACE_CROPS_DIR)}")