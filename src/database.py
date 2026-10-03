import sqlite3
import os
from datetime import datetime


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "student_database.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# CREATE TABLE
# ============================================================

def create_tables():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_id TEXT UNIQUE NOT NULL,

            student_name TEXT NOT NULL,

            department TEXT,

            study_time INTEGER,

            absences INTEGER,

            failures INTEGER,

            goout INTEGER,

            freetime INTEGER,

            health INTEGER,

            famrel INTEGER,

            traveltime INTEGER,

            predicted_score REAL,

            performance_level TEXT,

            early_warning TEXT,

            created_at TEXT

        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# INSERT STUDENT
# ============================================================

def add_student(
    student_id,
    student_name,
    department,
    study_time,
    absences,
    failures,
    goout,
    freetime,
    health,
    famrel,
    traveltime,
    predicted_score,
    performance_level,
    early_warning
):

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        cursor.execute(
            """
            INSERT INTO students (
                student_id,
                student_name,
                department,
                study_time,
                absences,
                failures,
                goout,
                freetime,
                health,
                famrel,
                traveltime,
                predicted_score,
                performance_level,
                early_warning,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (
                student_id,
                student_name,
                department,
                study_time,
                absences,
                failures,
                goout,
                freetime,
                health,
                famrel,
                traveltime,
                predicted_score,
                performance_level,
                early_warning,
                created_at
            )
        )

        connection.commit()

        success = True
        message = "Student saved successfully."

    except sqlite3.IntegrityError:

        success = False
        message = "Student ID already exists."

    except Exception as e:

        success = False
        message = str(e)

    connection.close()

    return success, message


# ============================================================
# GET ALL STUDENTS
# ============================================================

def get_students():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM students
        ORDER BY created_at DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# ============================================================
# GET ONE STUDENT
# ============================================================

def get_student(student_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM students
        WHERE student_id = ?
        """,
        (student_id,)
    )

    row = cursor.fetchone()

    connection.close()

    return row


# ============================================================
# DELETE STUDENT
# ============================================================

def delete_student(student_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM students
        WHERE student_id = ?
        """,
        (student_id,)
    )

    connection.commit()

    deleted = cursor.rowcount > 0

    connection.close()

    return deleted


# ============================================================
# DATABASE STATISTICS
# ============================================================

def get_statistics():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_students,
            AVG(predicted_score) AS average_score,
            AVG(absences) AS average_absences
        FROM students
        """
    )

    result = cursor.fetchone()

    connection.close()

    return result


# ============================================================
# INITIALIZE DATABASE
# ============================================================

create_tables()