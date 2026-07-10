import detector
import cv2
import numpy as np  

video_path = "vtest.avi"  
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    detected_objects = detector.detect_objects(frame, confidence_threshold=0.5)
    
   
    for i, obj in enumerate(detected_objects):
        print(f"Object {i+1}: Type={obj['type']}, Confidence={obj['confidence']:.2f}")
    
    
    # Show count
    total_objects = len(detected_objects)
    cv2.putText(frame, f"Detected: {total_objects} objects", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Object Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total frames: {frame_count}")