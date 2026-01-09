# Model/location_model.py
import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "visionmate.db")

class LocationModel:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        cursor = self.conn.cursor()
        
        # Tạo bảng với đầy đủ cột cần thiết
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                device TEXT,
                address TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Kiểm tra và thêm cột device nếu chưa có
        cursor.execute("PRAGMA table_info(location)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'device' not in columns:
            cursor.execute("ALTER TABLE location ADD COLUMN device TEXT")
        
        self.conn.commit()

    def insert(self, lat, lng, device=None, address=None):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO location (lat, lng, device, address) VALUES (?, ?, ?, ?)",
            (lat, lng, device, address)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_latest(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT lat, lng, device, address, timestamp FROM location ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            return {
                'lat': float(row[0]) if row[0] is not None else 0.0,
                'lng': float(row[1]) if row[1] is not None else 0.0,
                'device': row[2] or 'Unknown Device',
                'address': row[3] or '',
                'timestamp': row[4] or datetime.now().isoformat()
            }
        return None

    def get_history(self, limit=50):
        cursor = self.conn.cursor()
        cursor.execute("SELECT lat, lng, device, address, timestamp FROM location ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        locations = []
        for row in rows:
            try:
                locations.append({
                    'lat': float(row[0]) if row[0] is not None else 0.0,
                    'lng': float(row[1]) if row[1] is not None else 0.0,
                    'device': row[2] or 'Unknown Device',
                    'address': row[3] or '',
                    'timestamp': row[4] or datetime.now().isoformat()
                })
            except:
                continue
        
        return locations

    def clear_history(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM location")
        self.conn.commit()
        return cursor.rowcount

    def __del__(self):
        if hasattr(self, "conn"):
            self.conn.close()

location_model = LocationModel()