import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

# ==============================
# DATABASE PATH (default)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "visionmate.db")


class User:
    def __init__(self, row=None):
        self.userID = None
        self.name = None
        self.hashpw = None
        self.email = None
        self.phone = None
        self.address = None

        if row:
            self.read_row(row)

    def read_row(self, data: tuple):
        if not data:
            return

        # Note: schema order: id, name, hashpw, email, phone, address
        self.userID = data[0]
        self.name = data[1]
        self.hashpw = data[2]
        self.email = data[3]
        self.phone = data[4]
        self.address = data[5]

    def to_dict(self):
        return {
            "userID": self.userID,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address
        }

    def check_pw(self, pw: str) -> bool:
        if not self.hashpw:
            return False
        return check_password_hash(self.hashpw, pw)


class UserModel:
    """
    UserModel manages its own SQLite connection.
    Instantiate per-request (or per-thread) and close when done.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        # create a dedicated connection for this instance
        # Because we'll create one instance per request, default check_same_thread=True is fine.
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = None
        self._ensure_table()

    def _ensure_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                hashpw TEXT,
                email TEXT UNIQUE,
                phone TEXT,
                address TEXT
            )
        """)
        self.conn.commit()

    # ==============================
    # SIGNUP (create user)
    # ==============================
    def add(self, name: str, password_plain: str, email: str, phone: str = None, address: str = None):
        cursor = self.conn.cursor()
        try:
            hashpw = generate_password_hash(password_plain)
            cursor.execute(
                "INSERT INTO user(name, hashpw, email, phone, address) VALUES (?, ?, ?, ?, ?)",
                (name, hashpw, email, phone, address)
            )
            self.conn.commit()
            return cursor.lastrowid

        except sqlite3.IntegrityError:
            raise Exception("Email đã tồn tại!")
        except Exception as e:
            raise Exception(f"Lỗi khi thêm user: {str(e)}")

    # ==============================
    # LOGIN (authenticate user)
    # ==============================
    def auth(self, email, pw):
        user = self.get_by_email(email)
        if not user:
            return False
        return user.check_pw(pw)

    # ==============================
    # GET USER
    # ==============================
    def get_by_email(self, email):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM user WHERE email = ?", (email,))
        row = cursor.fetchone()
        return User(row=row) if row else None

    def get_by_id(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM user WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return User(row=row) if row else None

    def get_all_users(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, email, phone, address FROM user")
        rows = cursor.fetchall()
        # return list of dicts
        users = []
        for r in rows:
            users.append({
                "userID": r[0],
                "name": r[1],
                "email": r[2],
                "phone": r[3],
                "address": r[4]
            })
        return users

    # ==============================
    # UPDATE USER PROFILE
    # ==============================
    def update_user(self, user_id, name=None, email=None, phone=None, address=None):
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE user SET
                name = ?,
                email = ?,
                phone = ?,
                address = ?
            WHERE id = ?
        """, (name, email, phone, address, user_id))
        self.conn.commit()
        return True

    # ==============================
    # DELETE USER
    # ==============================
    def delete_user(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM user WHERE id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self):
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
