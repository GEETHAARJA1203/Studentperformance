import hashlib
import sqlite3
import os
from datetime import datetime


# ---------------------------------------------------------
# DATABASE PATH
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATABASE_PATH = os.path.join(BASE_DIR, "student_database.db")


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


# ---------------------------------------------------------
# PASSWORD HASHING
# ---------------------------------------------------------

def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# CREATE USERS TABLE
# ---------------------------------------------------------

def create_users_table():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            student_id TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# CREATE USER
# ---------------------------------------------------------

def create_user(
    username,
    password,
    role,
    student_id=None
):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                password_hash,
                role,
                student_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                hash_password(password),
                role,
                student_id,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        conn.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        conn.close()


# ---------------------------------------------------------
# AUTHENTICATE USER
# ---------------------------------------------------------

def authenticate_user(username, password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            username,
            password_hash,
            role,
            student_id
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    user = cursor.fetchone()

    conn.close()

    if user is None:
        return None

    stored_hash = user[2]

    if hash_password(password) == stored_hash:

        return {
            "id": user[0],
            "username": user[1],
            "role": user[3],
            "student_id": user[4]
        }

    return None


# ---------------------------------------------------------
# DEFAULT DEMO USERS
# ---------------------------------------------------------

def create_default_users():

    create_users_table()

    # Admin
    create_user(
        username="admin",
        password="admin123",
        role="Admin"
    )

    # Faculty
    create_user(
        username="faculty",
        password="faculty123",
        role="Faculty"
    )

    # Student
    create_user(
        username="student",
        password="student123",
        role="Student",
        student_id="STU001"
    )


# ---------------------------------------------------------
# INITIALIZE AUTHENTICATION
# ---------------------------------------------------------

create_default_users()