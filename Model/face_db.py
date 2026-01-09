import os
import sqlite3
import json
import traceback
from typing import List, Optional, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "visionmate.db")

print(f"[FACE_DB] Database path: {DB_PATH}")

def db_conn():
    """Tạo kết nối database"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"[FACE_DB] Connection error: {e}")
        raise

def ensure_table():
    """Tạo bảng nếu chưa tồn tại"""
    try:
        conn = db_conn()
        cur = conn.cursor()
        
        # Tạo bảng faces
        cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                faceID INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                relationship TEXT,
                email TEXT,
                image_path TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at DATETIME DEFAULT (datetime('now','localtime'))
            )
        """)
        
        # Tạo index
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_created ON faces(created_at DESC)")
        
        conn.commit()
        conn.close()
        print("[FACE_DB] Table ensured")
        
    except Exception as e:
        print(f"[FACE_DB] ensure_table error: {e}")

def insert_face(name: str, relationship: str, email: Optional[str], image_path: str, embedding):
    """Thêm khuôn mặt mới vào database với sync cache"""
    try:
        conn = db_conn()
        cur = conn.cursor()
        
        print(f"[FACE_DB] Inserting face: {name}, image: {image_path}")
        
        # Chuyển embedding về JSON
        try:
            if hasattr(embedding, 'tolist'):
                emb_list = embedding.tolist()
            elif isinstance(embedding, list):
                emb_list = embedding
            else:
                emb_list = list(embedding)
            
            emb_json = json.dumps(emb_list)
        except Exception as e:
            print(f"[FACE_DB] Embedding conversion error: {e}")
            raise Exception(f"Lỗi chuyển đổi embedding: {e}")
        
        # Kiểm tra xem khuôn mặt đã tồn tạị trong ảnh chưa
        cur.execute(
            "SELECT faceID FROM faces WHERE name = ? AND image_path = ?",
            (name, image_path)
        )
        existing = cur.fetchone()
         
        if existing:
            print(f"[FACE_DB] Face already exists: {name}")
            conn.close()
            return False
        
        # Thêm mới
        cur.execute(
            """INSERT INTO faces (name, relationship, email, image_path, embedding) 
               VALUES (?, ?, ?, ?, ?)""",
            (name, relationship, email, image_path, emb_json)
        )
        
        conn.commit()
        conn.close()
        
        print(f"[FACE_DB] Face inserted: {name}")
        
        # SYNC CACHE NGAY SAU KHI INSERT
        try:
            from .face_model import sync_faces_cache
            sync_faces_cache()
            print("[FACE_DB] Cache synced after insertion")
        except ImportError as e:
            print(f"[FACE_DB] Cannot import face_model to sync cache: {e}")
        
        return True
        
    except sqlite3.Error as e:
        print(f"❌ [FACE_DB] Database error: {e}")
        return False
    except Exception as e:
        print(f"[FACE_DB] Insert error: {e}")
        return False

def get_all_faces() -> List[Dict]:
    """Lấy tất cả khuôn mặt"""
    try:
        conn = db_conn()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT faceID, name, relationship, email, image_path, embedding, created_at 
            FROM faces 
            ORDER BY created_at DESC
        """)
        
        rows = cur.fetchall()
        conn.close()
        
        faces = []
        for r in rows:
            try:
                face_data = {
                    "id": r["faceID"],
                    "name": r["name"],
                    "relationship": r["relationship"] or "",
                    "email": r["email"] or "",
                    "image_path": r["image_path"],
                    "embedding": json.loads(r["embedding"]) if r["embedding"] else None,
                    "created_at": r["created_at"]
                }
                faces.append(face_data)
            except Exception as e:
                print(f"[FACE_DB] Error parsing face {r['faceID']}: {e}")
                continue
        
        print(f"[FACE_DB] Loaded {len(faces)} faces")
        return faces
        
    except Exception as e:
        print(f"[FACE_DB] get_all_faces error: {e}")
        return []

def get_face_by_id(face_id: int):
    """Lấy khuôn mặt theo ID"""
    try:
        conn = db_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT * FROM faces WHERE faceID = ?", (face_id,))
        row = cur.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row["faceID"],
                "name": row["name"],
                "relationship": row["relationship"],
                "email": row["email"],
                "image_path": row["image_path"],
                "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
                "created_at": row["created_at"]
            }
        return None
        
    except Exception as e:
        print(f"[FACE_DB] get_face_by_id error: {e}")
        return None

def delete_face(face_id: int) -> bool:
    """Xóa khuôn mặt và file ảnh với sync cache"""
    try:
        print(f"[FACE_DB] Deleting face ID: {face_id}")
        
        conn = db_conn()
        cur = conn.cursor()
        
        # Lấy thông tin trước khi xóa
        cur.execute("SELECT name, image_path FROM faces WHERE faceID = ?", (face_id,))
        row = cur.fetchone()
        
        if not row:
            print(f"[FACE_DB] Face not found: {face_id}")
            conn.close()
            return False
        
        face_name = row["name"]
        image_path = row["image_path"]
        
        # Xóa từ database
        cur.execute("DELETE FROM faces WHERE faceID = ?", (face_id,))
        deleted_rows = cur.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_rows > 0:
            print(f"[FACE_DB] Database record deleted: {face_id}")
            
            # Xóa file ảnh nếu tồn tại
            if image_path:
                try:
                    # Tìm file trong các thư mục
                    possible_paths = [
                        os.path.join(PROJECT_ROOT, "uploads", image_path),
                        os.path.join(PROJECT_ROOT, "uploads", "face_crops", image_path),
                        image_path if os.path.exists(image_path) else None
                    ]
                    
                    for img_path in possible_paths:
                        if img_path and os.path.exists(img_path):
                            os.remove(img_path)
                            print(f"[FACE_DB] Image file deleted: {img_path}")
                            break
                except Exception as img_error:
                    print(f"[FACE_DB] Could not delete image: {img_error}")
            
            # SYNC CACHE NGAY SAU KHI XÓA
            try:
                from .face_model import sync_faces_cache
                sync_faces_cache()
                print("[FACE_DB] Cache synced after deletion")
            except ImportError as e:
                print(f"[FACE_DB] Cannot import face_model to sync cache: {e}")
            
            print(f"[FACE_DB] Successfully deleted face: {face_name} (ID: {face_id})")
            return True
        else:
            print(f"[FACE_DB] No face deleted")
            return False
            
    except sqlite3.Error as e:
        print(f"[FACE_DB] Database error: {e}")
        return False
    except Exception as e:
        print(f"[FACE_DB] delete_face error: {e}")
        traceback.print_exc()
        return False

def search_faces_by_name(name_query: str) -> List[Dict]:
    """Tìm khuôn mặt theo tên"""
    try:
        conn = db_conn()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT faceID, name, relationship, email, image_path, created_at 
            FROM faces 
            WHERE name LIKE ? 
            ORDER BY created_at DESC
        """, (f"%{name_query}%",))
        
        rows = cur.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]
        
    except Exception as e:
        print(f"[FACE_DB] search_faces_by_name error: {e}")
        return []

def count_faces() -> int:
    """Đếm tổng số khuôn mặt"""
    try:
        conn = db_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) as count FROM faces")
        result = cur.fetchone()
        conn.close()
        
        return result["count"] if result else 0
        
    except Exception as e:
        print(f"[FACE_DB] count_faces error: {e}")
        return 0

def sync_cache_after_operation():
    """Đồng bộ cache sau khi thao tác"""
    try:
        from .face_model import sync_faces_cache
        sync_faces_cache()
        return True
    except Exception as e:
        print(f"[FACE_DB] Cache sync failed: {e}")
        return False

# Khởi tạo bảng khi import
ensure_table()
print("[FACE_DB] Module initialized with cache sync support")