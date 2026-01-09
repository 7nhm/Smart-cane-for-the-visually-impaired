import sqlite3
import contextlib
from typing import List, Tuple, Any, Optional

class Database:
    def __init__(self, db_path: str = 'visionmate.db'):
        self.db_path = db_path
    
    @contextlib.contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def execute(self, query: str, params: Tuple = ()) -> sqlite3.Cursor:
        """Execute a single query"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
    
    def fetch_all(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Fetch all results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def fetch_one(self, query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        """Fetch single result"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.fetch_one(query, (table_name,))
        return result is not None

# Global database instance
db = Database()

if __name__ == '__main__':
    # Test database connection
    print("Testing database connection...")
    tables = db.fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
    print("Tables in database:")
    for table in tables:
        print(f" - {table['name']}")
        