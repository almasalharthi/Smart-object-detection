from ultralytics import YOLO
import cv2
from collections import Counter 
import time 
import os 

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

prev_time = 0
os.makedirs("screenshots", exist_ok=True) 
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    # Count detected objects
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        detections.append(class_name)

    counts = Counter(detections)

    annotated_frame = results[0].plot()

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    y = 30

    for name, count in counts.items():
        cv2.putText(
            annotated_frame,
            f"{name}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
    )
    y += 30

    cv2.imshow("Smart Object Detector", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = f"screenshots/screenshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"Saved: {filename}")

    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

