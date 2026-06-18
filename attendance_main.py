import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import time
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta

# --- MySQL Database Configuration Core ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Kept completely blank for Homebrew MySQL setup
    'database': 'attendance_system'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"[DATABASE ERROR] Connection failed: {e}")
    return None

# --- Initialize Optimized MediaPipe Hand Tracking ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    model_complexity=0, 
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# --- Load Encodings Cache directly from MySQL ---
known_encodings = []
known_roll_nos = []
known_names = []

def reload_student_cache():
    global known_encodings, known_roll_nos, known_names
    known_encodings, known_roll_nos, known_names = [], [], []
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT roll_no, name, encoding FROM students")
        rows = cursor.fetchall()
        for row in rows:
            known_roll_nos.append(row[0])
            known_names.append(row[1])
            # Parse the string format back to a numpy float64 array map
            known_encodings.append(np.fromstring(row[2], sep=','))
        cursor.close()
        conn.close()

# Initial core download from relational database server
reload_student_cache()

# Variables for Performance & UI Coordination
last_action_time = 0
cooldown_seconds = 4  
status_msg = "System Active [MySQL Online]"
msg_color = (0, 255, 0)

frame_count = 0
face_locations, face_encodings, face_names, face_rolls = [], [], [], []

def save_new_student(roll_no, name, encoding):
    encoding_str = ','.join(map(str, encoding))
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            query = "INSERT INTO students (roll_no, name, encoding) VALUES (%s, %s, %s)"
            cursor.execute(query, (roll_no, name, encoding_str))
            conn.commit()
            cursor.close()
            conn.close()
            reload_student_cache() # Synchronize relational database memory layout instantly
            return True
        except Error as e:
            print(f"[SQL ERROR] Registration aborted: {e}")
    return False

def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    thumb_is_open = lm[4].y < lm[3].y
    index_is_open = lm[8].y < lm[6].y
    middle_is_open = lm[12].y < lm[10].y
    ring_is_open = lm[16].y < lm[14].y
    pinky_is_open = lm[20].y < lm[18].y

    if index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "FULL_PALM"
    if thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        if lm[4].y < lm[0].y:
            return "THUMBS_UP"
    return "UNKNOWN"

def get_opencv_input(prompt_text):
    """
    Captures live keyboard strokes using OpenCV buffers.
    Safely bypasses the broken, crash-prone macOS Tkinter UI thread layer.
    """
    user_input = ""
    while True:
        input_bg = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(input_bg, prompt_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(input_bg, user_input + "_", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(input_bg, "Press ENTER to confirm | ESC to cancel", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        cv2.imshow("Registration Entry", input_bg)
        key = cv2.waitKey(0)
        
        if key == 13: # Enter Key
            break
        elif key == 27: # Escape Key
            return None
        elif key == 127 or key == 8: # Backspace configurations
            user_input = user_input[:-1]
        elif 32 <= key <= 126: # Character keystroke constraints
            user_input += chr(key)
            
    cv2.destroyWindow("Registration Entry")
    return user_input.strip()

def process_attendance(roll_no, name):
    conn = get_db_connection()
    if not conn:
        return "DB Offline. Punch Denied.", (0, 0, 255)
        
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now()
    now_str = now_time.strftime("%H:%M:%S")
    
    try:
        cursor = conn.cursor(buffered=True)
        # Check for matching log row targets matching today's metric
        cursor.execute("SELECT id, in_time, out_time FROM attendance_logs WHERE log_date = %s AND roll_no = %s", (today, roll_no))
        record = cursor.fetchone()
        
        if not record:
            # Create a fresh registration record entry
            cursor.execute("INSERT INTO attendance_logs (log_date, roll_no, name, in_time) VALUES (%s, %s, %s, %s)", (today, roll_no, name, now_str))
            conn.commit()
            msg, color = f"In Logged: Welcome {name}!", (0, 255, 0)
        else:
            log_id, in_time_str, out_time_str = record
            
            if out_time_str != 'Pending':
                msg, color = f"{name} already processed today.", (0, 165, 255)
            else:
                # Normalize timedelta objects thrown by MySQL native time field parsers
                if isinstance(in_time_str, timedelta):
                    in_time_dt = datetime.combine(datetime.today(), datetime.min.time()) + in_time_str
                else:
                    in_time_dt = datetime.strptime(f"{today} {in_time_str}", "%Y-%m-%d %H:%M:%S")
                
                time_difference = now_time - in_time_dt
                if time_difference >= timedelta(hours=4):
                    cursor.execute("UPDATE attendance_logs SET out_time = %s WHERE id = %s", (now_str, log_id))
                    conn.commit()
                    msg, color = f"Out Logged: Goodbye {name}!", (255, 255, 0)
                else:
                    remaining = timedelta(hours=4) - time_difference
                    mins_left = int(remaining.total_seconds() / 60)
                    msg, color = f"Class Active. Locked for {mins_left}m", (0, 0, 255)
                    
        cursor.close()
        conn.close()
        return msg, color
    except Error as e:
        return f"Database Error: {e}", (0, 0, 255)

# --- Camera Stream Execution Loop ---
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Production MySQL Core Running. Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time()
    frame_count += 1

    # Frame skipping calculation optimization rules for Apple Silicon
    process_this_frame = (frame_count % 4 == 0)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb_frame)
    current_gesture = "UNKNOWN"
    
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_gesture = detect_gesture(hand_landmarks)

    if process_this_frame:
        # Scale array operations down to 0.25 (Fast processing dimension mapping)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        face_names = []
        face_rolls = []
        
        for face_encoding in face_encodings:
            name = "Unknown"
            roll_no = None
            
            if known_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        roll_no = known_roll_nos[best_match_index]
            
            face_names.append(name)
            face_rolls.append(roll_no)

    # Scale the coordinates back up by 4 to display full-size bounding rectangles
    for (top, right, bottom, left), name, roll_no in zip(face_locations, face_names, face_rolls):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        box_color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        
        if name == "Unknown":
            cv2.putText(frame, "Unregistered Profile", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if current_gesture == "FULL_PALM" and (current_time - last_action_time > cooldown_seconds):
                high_res_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
                if high_res_encoding:
                    new_name = get_opencv_input("Enter Student's Full Name:")
                    if new_name:
                        new_roll = get_opencv_input(f"Enter Roll Number for {new_name}:")
                    else:
                        new_roll = None
                    
                    if new_name and new_roll:
                        if save_new_student(new_roll, new_name, high_res_encoding[0]):
                            status_msg = f"Saved to SQL: Welcome {new_name}!"
                            msg_color = (0, 255, 0)
                        else:
                            status_msg = "Database write failed."
                            msg_color = (0, 0, 255)
                        face_locations, face_names, face_rolls = [], [], []
                        break
                    else:
                        status_msg = "Registration Cancelled."
                        msg_color = (0, 0, 255)
                last_action_time = time.time()
        else:
            cv2.putText(frame, f"{name} [{roll_no}]", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if current_gesture == "THUMBS_UP" and (current_time - last_action_time > cooldown_seconds):
                status_msg, msg_color = process_attendance(roll_no, name)
                last_action_time = time.time()

    # Graphic HUD UI Overlay Elements
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (20, 20, 20), -1)
    cv2.putText(frame, f"SYSTEM STATUS: {status_msg}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 2)
    cv2.putText(frame, f"Gesture Mode: {current_gesture}", (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    
    cv2.imshow('Biometric Attendance Core', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()