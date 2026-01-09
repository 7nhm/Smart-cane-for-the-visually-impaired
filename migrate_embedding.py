import sqlite3
import os

def migrate_embedding_column():
    """
    Migration script: Đảm bảo cột embedding_data tồn tại trong bảng faces
    """
    
    conn = sqlite3.connect('visionmate.db')
    cursor = conn.cursor()
    
    try:
        # Kiểm tra xem bảng faces có tồn tại không
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
        if not cursor.fetchone():
            print("❌ Table 'faces' doesn't exist. Run database.py first.")
            return
        
        # Kiểm tra cột embedding_data
        cursor.execute("PRAGMA table_info(faces)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'embedding_data' not in columns:
            print("🔄 Adding embedding_data column to faces table...")
            cursor.execute('ALTER TABLE faces ADD COLUMN embedding_data TEXT')
            conn.commit()
            print("✅ Added embedding_data column successfully!")
        else:
            print("✅ embedding_data column already exists - no migration needed")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    migrate_embedding_column()