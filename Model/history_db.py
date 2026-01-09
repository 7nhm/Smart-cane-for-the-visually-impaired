import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "visionmate.db")

class HistoryItem:
    def __init__(self, id, face_name, device, confidence, image, timestamp, is_cropped_face=0):
        self.id = id
        self.face_name = face_name
        self.device = device
        self.confidence = confidence
        self.image = image
        self.timestamp = timestamp
        self.is_cropped_face = is_cropped_face

def ensure_history_table():
    """Đảm bảo bảng history tồn tại với cột is_cropped_face"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Tạo bảng nếu chưa tồn tại
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_name TEXT,
            device TEXT,
            confidence REAL,
            image TEXT,
            timestamp TEXT,
            is_cropped_face INTEGER DEFAULT 0
        )
    """)
    
    # Kiểm tra và thêm cột is_cropped_face nếu chưa có
    try:
        cur.execute("SELECT is_cropped_face FROM face_history LIMIT 1")
    except sqlite3.OperationalError:
        cur.execute("ALTER TABLE face_history ADD COLUMN is_cropped_face INTEGER DEFAULT 0")
    
    conn.commit()
    conn.close()
    print("✅ History table checked/created")

def add_history(face_name, device, confidence, image, is_cropped_face=0):
    """Thêm bản ghi vào history"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO face_history (face_name, device, confidence, image, timestamp, is_cropped_face)
        VALUES (?, ?, ?, ?, datetime('now','localtime'), ?)
    """, (face_name, device, confidence, image, is_cropped_face))
    
    conn.commit()
    conn.close()
    return cur.lastrowid

def get_all_history(limit=500):
    """Lấy toàn bộ lịch sử"""
    ensure_history_table()
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, face_name, device, confidence, image, timestamp, is_cropped_face
        FROM face_history 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    
    rows = cur.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append(HistoryItem(
            id=row[0],
            face_name=row[1],
            device=row[2],
            confidence=row[3],
            image=row[4],
            timestamp=row[5],
            is_cropped_face=row[6]
        ))
    
    return history

def delete_history_by_id(history_id):
    """Xóa bản ghi history"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("DELETE FROM face_history WHERE id = ?", (history_id,))
    
    conn.commit()
    affected = cur.rowcount
    conn.close()
    
    return affected > 0

def clear_all_history():
    """Xóa toàn bộ lịch sử"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("DELETE FROM face_history")
    
    conn.commit()
    affected = cur.rowcount
    conn.close()
    
    return affected