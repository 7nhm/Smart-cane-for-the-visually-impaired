import sqlite3
import os
import threading
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "visionmate.db")

class Device:
    def __init__(self, row=None):
        self.deviceID = None
        self.device_name = None
        self.lat = None
        self.lng = None
        self.status = None
        self.sim_number = None
        self.last_heartbeat = None
        self.created_at = None
        if row:
            self.read_row(row)

    def read_row(self, row):
        self.deviceID = row[0]
        self.device_name = row[1]
        self.lat = row[2]
        self.lng = row[3]
        self.status = row[4] if row[4] else 'offline'
        self.sim_number = row[5]
        self.last_heartbeat = row[6] if len(row) > 6 else None
        self.created_at = row[7] if len(row) > 7 else None
        
    def __repr__(self):
        return f"Device(id={self.deviceID}, name={self.device_name}, status={self.status}, last_heartbeat={self.last_heartbeat})"

class DeviceModel:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_table()
        self._start_offline_checker()

    def _ensure_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                deviceID INTEGER PRIMARY KEY AUTOINCREMENT,
                device_name TEXT NOT NULL UNIQUE,
                lat REAL,
                lng REAL,
                status TEXT DEFAULT 'offline',
                sim_number TEXT,
                last_heartbeat TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add(self, device_name, sim_number):
        try:
            cur = self.conn.cursor()
            cur.execute(
                """INSERT INTO devices(device_name, sim_number, status) 
                   VALUES (?, ?, 'offline')""",
                (device_name, sim_number)
            )
            self.conn.commit()
            print(f"Added device: {device_name}")
            return cur.lastrowid
        except sqlite3.IntegrityError:
            print(f"Device {device_name} already exists")
            return None

    def get_all(self):
        try:
            rows = self.conn.execute("SELECT * FROM devices ORDER BY deviceID DESC").fetchall()
            devices = [Device(row) for row in rows]
            print(f"Total devices: {len(devices)}")
            return devices
        except Exception as e:
            print(f"Error getting devices: {e}")
            return []

    def get_by_id(self, device_id):
        try:
            row = self.conn.execute("SELECT * FROM devices WHERE deviceID = ?", (device_id,)).fetchone()
            if row:
                return Device(row)
            return None
        except Exception as e:
            print(f"Error getting device by ID: {e}")
            return None

    def get_by_name(self, device_name):
        try:
            row = self.conn.execute("SELECT * FROM devices WHERE device_name = ?", (device_name,)).fetchone()
            if row:
                return Device(row)
            return None
        except Exception as e:
            print(f"Error getting device by name: {e}")
            return None

    def delete(self, device_id):
        try:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM devices WHERE deviceID = ?", (device_id,))
            self.conn.commit()
            print(f"Deleted device ID: {device_id}")
            return cur.rowcount > 0
        except Exception as e:
            print(f"Error deleting device: {e}")
            return False

    def update_location(self, device_name, lat, lng):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE devices SET lat=?, lng=? WHERE device_name=?",
                (lat, lng, device_name)
            )
            self.conn.commit()
            if cur.rowcount > 0:
                print(f"Updated location for {device_name}: {lat}, {lng}")
                return True
            return False
        except Exception as e:
            print(f"Error updating location: {e}")
            return False

    def update_status(self, device_name, status):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE devices SET status=? WHERE device_name=?",
                (status, device_name)
            )
            self.conn.commit()
            if cur.rowcount > 0:
                print(f"Updated status for {device_name}: {status}")
                return True
            return False
        except Exception as e:
            print(f"Error updating status: {e}")
            return False

    def update_heartbeat(self, device_name, lat=None, lng=None):
        try:
            timestamp = datetime.now().isoformat()
            cur = self.conn.cursor()
            
            if lat is not None and lng is not None:
                # Update both location and heartbeat
                cur.execute(
                    """UPDATE devices 
                       SET status='online', 
                           last_heartbeat=?,
                           lat=?,
                           lng=?
                       WHERE device_name=?""",
                    (timestamp, lat, lng, device_name)
                )
            else:
                # Update only heartbeat
                cur.execute(
                    """UPDATE devices 
                       SET status='online', 
                           last_heartbeat=?
                       WHERE device_name=?""",
                    (timestamp, device_name)
                )
            
            # If device doesn't exist, create it
            if cur.rowcount == 0:
                cur.execute(
                    """INSERT INTO devices 
                       (device_name, status, last_heartbeat, lat, lng) 
                       VALUES (?, 'online', ?, ?, ?)""",
                    (device_name, timestamp, lat, lng)
                )
                print(f"Created new device: {device_name}")
            
            self.conn.commit()
            print(f"Heartbeat from {device_name} at {timestamp}")
            return True
            
        except Exception as e:
            print(f"Error updating heartbeat: {e}")
            return False

    def check_and_update_offline_devices(self):
        """Tự động cập nhật thiết bị offline nếu không có heartbeat trong 2 phút"""
        try:
            offline_threshold = 120  # 2 phút
            
            # Lấy tất cả thiết bị online
            online_devices = self.conn.execute(
                "SELECT * FROM devices WHERE status = 'online'"
            ).fetchall()
            
            now = datetime.now()
            updated_count = 0
            
            for device_row in online_devices:
                device = Device(device_row)
                if device.last_heartbeat:
                    try:
                        # Chuyển đổi last_heartbeat string thành datetime
                        last_heartbeat_dt = datetime.fromisoformat(device.last_heartbeat)
                        seconds_since_heartbeat = (now - last_heartbeat_dt).total_seconds()
                        
                        if seconds_since_heartbeat > offline_threshold:
                            # Chuyển sang offline
                            self.conn.execute(
                                "UPDATE devices SET status='offline' WHERE deviceID=?",
                                (device.deviceID,)
                            )
                            updated_count += 1
                            print(f"Auto-offline: {device.device_name} ({seconds_since_heartbeat:.0f}s no heartbeat)")
                            
                    except Exception as e:
                        print(f"Error parsing heartbeat for {device.device_name}: {e}")
                        # Nếu có lỗi parsing, chuyển sang offline
                        self.conn.execute(
                            "UPDATE devices SET status='offline' WHERE deviceID=?",
                            (device.deviceID,)
                        )
                        updated_count += 1
            
            self.conn.commit()
            if updated_count > 0:
                print(f"Auto-updated {updated_count} devices to offline")
            
            return True
            
        except Exception as e:
            print(f"Error in offline checker: {e}")
            return False

    def _start_offline_checker(self):
        """Khởi động thread kiểm tra thiết bị offline"""
        def check_offline_loop():
            while True:
                try:
                    self.check_and_update_offline_devices()
                    time.sleep(30)  # Kiểm tra mỗi 30 giây
                except Exception as e:
                    print(f"Error in offline checker loop: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=check_offline_loop, daemon=True)
        thread.start()
        print("Started offline checker thread")

# Tạo instance global
device_model = DeviceModel()