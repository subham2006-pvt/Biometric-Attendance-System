# Biometric Attendance System with Gesture Control 🎯🤖

A production-ready, high-performance facial recognition attendance management system engineered for macOS (optimized for Apple Silicon M-series architectures). The application processes live video feeds at 30 FPS using frame-skipping optimizations, tracks real-time physical gestures via MediaPipe, and coordinates tracking records directly into a localized relational MySQL database server.

---

## 🛠️ System Architecture & Technology Stack

* **Core Execution Engine:** Python 3
* **Computer Vision Pipeline:** OpenCV (Dynamic camera streaming layer)
* **Machine Learning Frame Processing:** MediaPipe (Hand Landmark Tracking)
* **Biometric Facial Vector Maps:** `face_recognition` (dlib multi-point facial array maps)
* **Database Management System:** MySQL Server (Homebrew Deployment)

---

## 🎮 Interface Gesture Control Guide

The user interface safely handles dynamic user interactions and registrations entirely inside an OpenCV custom overlay HUD loop to completely bypass traditional macOS thread rendering blocks:

| Gesture Input | System Interpretation Context | Executed Action Mapping |
| :--- | :--- | :--- |
| **Open Palm ✋** | Profile Registration Mode | Triggers high-resolution face vector calculations and opens terminal data panels to register a new user name and roll number directly into MySQL. |
| **Thumbs Up 👍** | Biometric Authentication Check-In | Matches active facial hashes against known database rows and commits timestamp-accurate Check-In or Check-Out entries. |

---

## 📂 Relational Database Architecture

The system coordinates relational records using an isolated localized database structure named `attendance_system`:

### 1. `students` Table Schema
Stores registered profiles along with compressed 128-dimension floating-point biometric coordinate maps transformed into string arrays.
* `roll_no` (VARCHAR, Primary Key)
* `name` (VARCHAR)
* `encoding` (TEXT)

### 2. `attendance_logs` Table Schema
Manages daily records with automated 4-hour active session lock protection constraints to prevent duplicate check-ins.
* `id` (INT, Auto-Increment Primary Key)
* `log_date` (DATE)
* `roll_no` (VARCHAR, Foreign Key referencing `students`)
* `name` (VARCHAR)
* `in_time` (TIME)
* `out_time` (VARCHAR, Defaults to `'Pending'`)

---

## 🚀 Installation & Local Environment Setup

### 1. Prerequisites (macOS Package Management)
Ensure you have Homebrew installed on your system, then spin up the relational database server:
```bash
brew install mysql
brew services start mysql