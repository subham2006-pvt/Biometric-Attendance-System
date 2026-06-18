import os
import cv2
import sys
import time
import numpy as np
import datetime
import threading
from flask import Flask, render_template, Response, jsonify, send_file
import mysql.connector
import mediapipe as mp
import face_recognition
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

app = Flask(__name__)

# --- DATABASE SETUP ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Add your MySQL password here if configured
    'database': 'attendance_system'
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# Ensure database tables exist at boot
try:
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        roll_no VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        encoding LONGTEXT NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        log_date DATE NOT NULL,
        roll_no VARCHAR(50) NOT NULL,
        name VARCHAR(100) NOT NULL,
        in_time TIME NOT NULL,
        out_time VARCHAR(50) DEFAULT 'Pending',
        FOREIGN KEY (roll_no) REFERENCES students(roll_no) ON DELETE CASCADE
    );
    """)
    db.commit()
    cursor.close()
    db.close()
    print("✓ Local MySQL Database Tables Validated Successfully.")
except Exception as e:
    print(f"✗ Database Initialization Failure: {e}")
    sys.exit(1)

# --- COMPUTER VISION INITIALIZATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Thread-safe Frame Buffer Object
class VideoCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

# Global hardware processing pointer
video_stream = VideoCaptureThread(src=0)

# --- CACHE AND BIOMETRIC MEMORY MANAGEMENT ---
known_encodings = []
known_roll_nos = []
known_names = []
last_gesture_time = 0
GESTURE_COOLDOWN = 1.5  # Seconds between physical interactions

def reload_biometric_cache():
    global known_encodings, known_roll_nos, known_names
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT roll_no, name, encoding FROM students")
        records = cursor.fetchall()
        
        encodings, rolls, names = [], [], []
        for row in records:
            # Reconstruct string array into 128-dimensional floating point numpy array
            arr = np.fromstring(row['encoding'].strip('[]'), sep=',')
            if len(arr) == 128:
                encodings.append(arr)
                rolls.append(row['roll_no'])
                names.append(row['name'])
        
        known_encodings, known_roll_nos, known_names = encodings, rolls, names
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error refreshing cache: {e}")

reload_biometric_cache()

# --- HARDWARE EVENT MATCHERS ---
def process_gesture_and_biometrics(rgb_frame, bgr_frame):
    global last_gesture_time
    current_time = time.time()
    if current_time - last_gesture_time < GESTURE_COOLDOWN:
        return

    # Process Hand Landmarks via MediaPipe
    results = hands.process(rgb_frame)
    if not results.multi_hand_landmarks:
        return

    for hand_landmarks in results.multi_hand_landmarks:
        lm = hand_landmarks.landmark
        
        # Test for Thumbs Up (👍) -> Check-In / Check-Out
        thumb_is_up = lm[4].y < lm[3].y < lm[2].y
        fingers_are_down = all(lm[i].y > lm[i-2].y for i in [8, 12, 16, 20])
        
        if thumb_is_up and fingers_are_down:
            print("👍 Thumbs Up Detected! Authenticating user profile...")
            last_gesture_time = current_time
            handle_authentication_event(bgr_frame)
            return

        # Test for Open Palm (✋) -> Profile Registration Trigger
        palm_is_open = all(lm[i].y < lm[i-2].y for i in [8, 12, 16, 20])
        if palm_is_open:
            print("✋ Open Palm Detected! Launching terminal profile registration wizard...")
            last_gesture_time = current_time
            handle_registration_event(bgr_frame)
            return

def handle_authentication_event(frame):
    face_locs = face_recognition.face_locations(frame)
    face_encs = face_recognition.face_encodings(frame, face_locs)
    
    if not face_encs:
        print("✗ Authentication Error: Face footprint not found in frame layout.")
        return

    for enc in face_encs:
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        if True in matches:
            idx = matches.index(True)
            roll_no = known_roll_nos[idx]
            name = known_names[idx]
            execute_attendance_sql(roll_no, name)
            return
    print("✗ Authentication Error: Target face not recognized against system records.")

def execute_attendance_sql(roll_no, name):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        today = datetime.date.today().strftime('%Y-%m-%d')
        now_time = datetime.datetime.now().strftime('%H:%M:%S')

        # Check for active existing logs today
        cursor.execute(
            "SELECT id, in_time, out_time FROM attendance_logs WHERE roll_no = %s AND log_date = %s ORDER BY id DESC LIMIT 1",
            (roll_no, today)
        )
        log = cursor.fetchone()

        if not log:
            # Insert direct brand-new Clock-In registry
            cursor.execute(
                "INSERT INTO attendance_logs (log_date, roll_no, name, in_time) VALUES (%s, %s, %s, %s)",
                (today, roll_no, name, now_time)
            )
            print(f"✓ Session Clocked In Successfully: {name} [{roll_no}]")
        else:
            if log['out_time'] != 'Pending':
                print(f"ℹ Log Guard alert: {name} already has a fully completed shift layout today.")
            else:
                # Enforce rigid 4-hour active session lock protection constraint
                in_datetime = datetime.datetime.strptime(str(log['in_time']), '%H:%M:%S')
                now_datetime = datetime.datetime.strptime(now_time, '%H:%M:%S')
                elapsed_hours = (now_datetime - in_datetime).total_seconds() / 3600.0

                if elapsed_hours < 4.0:
                    print(f"⚠️ Lockout Warning: Multi-punch prevented. Shift lockout active for next {4.0 - elapsed_hours:.2f} hours.")
                else:
                    cursor.execute(
                        "UPDATE attendance_logs SET out_time = %s WHERE id = %s",
                        (now_time, log['id'])
                    )
                    print(f"✓ Session Clocked Out Successfully: {name} [{roll_no}]")
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"SQL Error during processing: {e}")

def handle_registration_event(frame):
    face_locs = face_recognition.face_locations(frame)
    face_encs = face_recognition.face_encodings(frame, face_locs)

    if not face_encs:
        print("✗ Registration Error: Hold face steady within camera line of sight.")
        return

    target_encoding = face_encs[0]
    
    # Intercept processing thread loop to open terminal entry configuration safely
    print("\n--- NEW BIOMETRIC PROFILE ENROLLMENT WIZARD ---")
    reg_name = input("Enter Student Full Legal Name: ").strip()
    reg_roll = input("Enter Student Unique Roll Number: ").strip()

    if not reg_name or not reg_roll:
        print("✗ Registration Error: Name and Roll attributes cannot be saved empty.")
        return

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        encoding_str = np.array2string(target_encoding, separator=',', max_line_width=3000)
        
        cursor.execute(
            "INSERT INTO students (roll_no, name, encoding) VALUES (%s, %s, %s)",
            (reg_roll, reg_name, encoding_str)
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✓ Profile Linked Successfully into Core Memory Engine: {reg_name}")
        reload_biometric_cache()
    except mysql.connector.Error as err:
        print(f"✗ Local Database Insertion Collision Error: {err}")

# --- FLASK STREAM GENERATOR LOOP ---
def generate_camera_frames():
    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        # Render tracking metrics on the frame array
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        process_gesture_and_biometrics(rgb_frame, frame)

        # Encode processed frame to pass seamlessly through the dashboard viewport
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROUTING WEB SYSTEM ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/attendance_status')
def attendance_status():
    try:
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 1. Poll unique entries present today
        cursor.execute(
            "SELECT COUNT(DISTINCT roll_no) as total FROM attendance_logs WHERE log_date = %s", 
            (today_str,)
        )
        count_data = cursor.fetchone()
        present_count = count_data['total'] if count_data else 0

        # 2. Extract recent transactional logs for timeline
        cursor.execute(
            "SELECT name, roll_no, in_time, out_time FROM attendance_logs WHERE log_date = %s ORDER BY id DESC LIMIT 10", 
            (today_str,)
        )
        recent_logs = cursor.fetchall()
        cursor.close()
        conn.close()

        formatted_logs = []
        for log in recent_logs:
            in_time_str = str(log['in_time'])
            out_time_str = str(log['out_time'])
            formatted_logs.append({
                'name': log['name'],
                'roll_no': log['roll_no'],
                'in_time': in_time_str[:5],
                'out_time': out_time_str[:5] if out_time_str != 'Pending' else 'Pending'
            })

        return jsonify({
            'status': 'success',
            'present_count': present_count,
            'logs': formatted_logs
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/export_excel')
def export_excel():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT log_date, roll_no, name, in_time, out_time FROM attendance_logs ORDER BY id DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        wb = Workbook()
        ws = wb.active
        ws.title = "System Master Ledger"
        ws.views.sheetView[0].showGridLines = True

        headers = ["Log Date", "Roll Number", "Full Legal Profile Name", "Clock In Time", "Clock Out Status"]
        ws.append(headers)

        # Style Sheets Definitions (Classic Slate Blue)
        font_family = "Segoe UI"
        header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
        header_font = Font(name=font_family, size=11, bold=True, color="FFFFFF")
        even_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
        odd_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
        active_session_fill = PatternFill(start_color="F0FDF4", end_color="F0FDF4", fill_type="solid")
        
        thin_border = Border(
            left=Side(style='thin', color='E2E8F0'),
            right=Side(style='thin', color='E2E8F0'),
            top=Side(style='thin', color='E2E8F0'),
            bottom=Side(style='thin', color='E2E8F0')
        )

        for col_idx, text in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        for row_idx, data in enumerate(rows, 2):
            row_data = [str(data['log_date']), data['roll_no'], data['name'], str(data['in_time']), data['out_time']]
            ws.append(row_data)
            
            # Contextual Row Highlighting
            current_fill = active_session_fill if data['out_time'] == 'Pending' else (even_fill if row_idx % 2 == 0 else odd_fill)
            
            for col_idx in range(1, 6):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.font = Font(name=font_family, size=10)
                cell.fill = current_fill
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="left" if col_idx == 3 else "center", vertical="center")

        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        output_filename = "Biometric_Attendance_Ledger.xlsx"
        wb.save(output_filename)
        return send_file(output_filename, as_attachment=True)
    except Exception as e:
        return f"Operational Generation Timeout Exception: {e}", 500

if __name__ == '__main__':
    # Using Port 8000 to prevent port assignment blocks caused by native macOS services
    app.run(host='127.0.0.1', port=8000, debug=False)