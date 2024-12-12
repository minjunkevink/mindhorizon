import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mindhorizon.db")

def init_db():
    """Initialize the database tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # create Users table if not exists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );
    """)
    
    # create Metrics table if not exists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        hours_studied REAL,
        previous_scores REAL,
        sleep_hours REAL,
        sample_questions REAL,
        extracurricular INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES Users(id)
    );
    """)
    
    conn.commit()
    conn.close()

def create_user(username, password):
    """Create a new user with a plaintext password."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO Users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_username(username):
    """Retrieve a user record by username."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, username, password FROM Users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row:
        # row is (id, username, password)
        return {"id": row[0], "username": row[1], "password": row[2]}
    return None

def check_credentials(username, password):
    """Check if the given username and password match a user (plaintext)."""
    user = get_user_by_username(username)
    if user and user["password"] == password:
        return user["id"]
    return None

def insert_metrics(user_id, hours, prev_scores, sleep, sample_qs, extra):
    """Insert a new metrics record for a given user."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO Metrics (user_id, hours_studied, previous_scores, sleep_hours, sample_questions, extracurricular)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, hours, prev_scores, sleep, sample_qs, extra))
    conn.commit()
    conn.close()

def get_user_metrics(user_id):
    """Retrieve all metrics entries for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT hours_studied, previous_scores, sleep_hours, sample_questions, extracurricular, timestamp
        FROM Metrics
        WHERE user_id = ?
        """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_all_metrics():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT hours_studied, previous_scores, sleep_hours, sample_questions, extracurricular
        FROM Metrics
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database initialized.")