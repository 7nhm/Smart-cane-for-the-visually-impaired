import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../visionmate.db")


def ensure_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alert (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            message TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_alert(title, message):
    ensure_table()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO alert (title, message, timestamp)
        VALUES (?, ?, ?)
    """, (title, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()


# LẤY 10 TRẠNG THÁI MỚI NHẤT
def get_recent_alerts(limit=10):
    ensure_table()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT title, message, timestamp
        FROM alert
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()

    alerts = []

    for row in rows:
        alerts.append({
            "title": row[0],
            "message": row[1],
            "timestamp": row[2]
        })

    return alerts
