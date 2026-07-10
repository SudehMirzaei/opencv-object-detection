import cv2
import numpy as np

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt', 
    'MobileNetSSD_deploy.caffemodel'
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect_objects(frame, confidence_threshold=0.5):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            
            if class_id != 0:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                objects.append({
                    'type' : CLASSES[class_id],
                    'bbox' : (startX, startY, endX, endY),
                    'confidence' : confidence
                })
    
    return objects

