# database.py
import sqlite3
from datetime import datetime

DB_PATH = "attendance.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_name TEXT UNIQUE NOT NULL,
            matric_no TEXT,
            name TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            course TEXT,
            date TEXT,
            time_in TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    conn.commit()
    conn.close()
    print("[INFO] Database initialized.")

def add_student(folder_name, matric_no=None, name=None):
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (folder_name, matric_no, name) VALUES (?, ?, ?)",
                  (folder_name, matric_no, name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_student_by_folder(folder_name):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, folder_name, matric_no, name FROM students WHERE folder_name=?", (folder_name,))
    row = c.fetchone()
    conn.close()
    return row

def get_student_by_name(folder_name):
    return get_student_by_folder(folder_name)

def log_attendance_by_folder(folder_name, course=None):
    # find student id
    student = get_student_by_folder(folder_name)
    if not student:
        return False, "student not found"
    student_id = student[0]
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_in = now.strftime("%H:%M:%S")
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO attendance (student_id, course, date, time_in) VALUES (?, ?, ?, ?)",
              (student_id, course, date, time_in))
    conn.commit()
    conn.close()
    return True, {"student_id": student_id, "date": date, "time_in": time_in}

def get_attendance_report(course=None, date=None):
    conn = get_conn()
    c = conn.cursor()
    query = '''
        SELECT a.id, s.folder_name, s.matric_no, s.name, a.course, a.date, a.time_in
        FROM attendance a
        JOIN students s ON a.student_id = s.id
    '''
    params = []
    conds = []
    if course:
        conds.append("a.course = ?")
        params.append(course)
    if date:
        conds.append("a.date = ?")
        params.append(date)
    if conds:
        query += " WHERE " + " AND ".join(conds)
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
