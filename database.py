import sqlite3
import os
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "visionmate.db")

print(f"📊 Creating database at: {DB_PATH}")

# Xóa database cũ nếu có
if os.path.exists(DB_PATH):
    print("⚠️ Removing old database...")
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. Tạo bảng user
cursor.execute("""
    CREATE TABLE user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        hashpw TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        address TEXT
    )
""")
print("✅ Created 'user' table")

# 2. Tạo bảng face
cursor.execute("""
    CREATE TABLE face (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        relationship TEXT,
        email TEXT,
        image_path TEXT,
        embedding TEXT
    )
""")
print("✅ Created 'face' table")

# 3. Tạo bảng device
cursor.execute("""
    CREATE TABLE device (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_name TEXT,
        sim_number TEXT,
        status TEXT DEFAULT 'offline',
        last_seen TIMESTAMP,
        latitude REAL,
        longitude REAL
    )
""")
print("✅ Created 'device' table")

# 4. Tạo bảng history
cursor.execute("""
    CREATE TABLE history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        source TEXT,
        confidence REAL,
        image_path TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
print("✅ Created 'history' table")

# 5. Tạo bảng alert
cursor.execute("""
    CREATE TABLE alert (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        message TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
print("✅ Created 'alert' table")

# 6. Tạo user admin mẫu
admin_password = generate_password_hash("admin123")
cursor.execute(
    "INSERT INTO user (name, hashpw, email, phone, address) VALUES (?, ?, ?, ?, ?)",
    ("Admin", admin_password, "admin@visionmate.com", "0123456789", "Admin Address")
)
print("✅ Created admin user: admin@visionmate.com / admin123")

conn.commit()
conn.close()

print(f"\n✅ Database created successfully at: {os.path.abspath(DB_PATH)}")
print("📊 Database structure ready!")