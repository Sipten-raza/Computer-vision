from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

# Initialize YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Load the video
video_source = "dataset_video.mp4"
video = cv2.VideoCapture(video_source)

# Validate video load
if not video.isOpened():
    print("Failed to load video.")
    exit()

frame_idx = 0
crowd_history = []
crowd_events = []

# Helper: Measure distance
def are_near(p1, p2, limit=65):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < limit

# Helper: Identify grouped individuals
def has_crowd(people):
    clusters = []
    marked = set()

    for idx1, person1 in enumerate(people):
        if idx1 in marked:
            continue
        group = [person1]
        marked.add(idx1)
        for idx2, person2 in enumerate(people):
            if idx2 != idx1 and idx2 not in marked and are_near(person1, person2):
                group.append(person2)
                marked.add(idx2)
        if len(group) >= 3:
            clusters.append(group)

    return len(clusters) > 0

# Frame-by-frame analysis
while True:
    success, img = video.read()
    if not success:
        break

    detections = yolo_model(img)[0]
    detected_people = []

    for obj in detections.boxes:
        class_id = int(obj.cls[0])
        if yolo_model.names[class_id] == 'person':
            x1, y1, x2, y2 = obj.xyxy[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            detected_people.append((center_x, center_y))

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(img, (center_x, center_y), 4, (0, 0, 255), -1)

    # Apply crowd detection logic
    if has_crowd(detected_people):
        crowd_history.append((frame_idx, len(detected_people)))
        if len(crowd_history) >= 10:
            crowd_events.append(crowd_history[0])
            crowd_history.pop(0)
            print(f"👥 Crowd detected in frame {frame_idx} with {len(detected_people)} people.")
    else:
        crowd_history.clear()

    # Show real-time frame
    cv2.imshow("Crowd Monitor", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

video.release()
cv2.destroyAllWindows()

# Save results
results_df = pd.DataFrame(crowd_events, columns=["Frame No", "People Count"])
results_df.to_csv("crowd_log.csv", index=False)

print(f"CSV saved: crowd_log.csv")
print(f"Total frames: {frame_idx}")
print(f"Total crowd events: {len(crowd_events)}")
