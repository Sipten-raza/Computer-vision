# Task 3: Crowd Detection - RAI AI Engineer Internship

## 👨‍💻 Objective

Detect persons in a video using YOLOv8 and identify **crowd events**, defined as:
- A group of **3 or more persons** standing **close together** (within a certain distance),
- Persisting for **10 or more consecutive frames**.

## 📦 Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Ultralytics YOLOv8

## 🎯 How It Works

1. **Person Detection:**  
   YOLOv8 detects persons in each video frame and extracts their center points.

2. **Crowd Logic:**  
   A crowd is a group of ≥3 people where all members are closer than 50 pixels from at least one other member.  
   If such a group is detected for 10 continuous frames, it is logged as a **crowd event**.

3. **Logging Results:**  
   Crowd events are stored in a CSV file `crowd_log.csv` with:
   - Frame Number
   - Person Count in Crowd

4. **Live Visualization:**  
   Each frame displays:
   - `Frame Number`
   - `"👥 Crowd Detected"` message when crowd is present

## 📂 Folder Structure

├── detect_crowd.py # Main script
├── input_video.mp4 # Sample video used
├── crowd_log.csv # Output CSV with crowd events
├── requirements.txt # Dependencies
└── README.md # This file