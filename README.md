# Real-Time Object Detection with MobileNet-SSD

A real-time object detection system using OpenCV's DNN module with a pre-trained MobileNet-SSD model.

## Overview

This project performs real-time object detection on video files using a MobileNet-SSD neural network. It detects 20 object classes, including people, vehicles, animals, and everyday objects.

## How It Works

### Model Architecture

The project uses a MobileNet-SSD model trained with the Caffe framework and loaded using OpenCV's dnn.readNetFromCaffe() function.

The model consists of two required files:

1. MobileNetSSD_deploy.prototxt
   - Defines the network architecture.
   - Specifies layer types and configurations.
   - Describes connections between layers.
   - Defines the input and output specifications.

2. MobileNetSSD_deploy.caffemodel
   - Contains the trained weights learned during training on the Pascal VOC dataset.

## Object Classes

The model can detect the following 20 object classes (plus the background class):

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]
## Detection Pipeline

### 1. Frame Preprocessing

Each video frame is resized to 300 × 300 pixels, the required input size for MobileNet-SSD. The frame is converted into a blob and normalized before being passed to the network.

### 2. Forward Pass

The preprocessed frame is passed through the neural network using OpenCV's DNN module, which performs object detection.

### 3. Output Processing

The network returns a 4-dimensional tensor with the shape:

(1, 1, 100, 7)
Where:

- Dimension 0: Batch size (1 image)
- Dimension 1: Reserved dimension
- Dimension 2: Up to 100 detections
- Dimension 3: Detection data consisting of seven values:

| Index | Description |
|-------:|-------------|
| 0 | Batch ID |
| 1 | Class ID |
| 2 | Confidence score |
| 3 | Bounding box x_min |
| 4 | Bounding box y_min |
| 5 | Bounding box x_max |
| 6 | Bounding box y_max |

### Example Detection

[0, 7, 0.95, 0.20, 0.30, 0.50, 0.60]
This corresponds to:

- Class ID 7 → car
- Confidence: 95%
- Bounding box:
  - x_min = 0.20
  - y_min = 0.30
  - x_max = 0.50
  - y_max = 0.60

### 4. Filtering

Only detections with a confidence score greater than 50% are kept. Each valid detection is stored with:

- Class label
- Bounding box coordinates
- Confidence score

## Requirements

- Python 3.6 or later
- OpenCV (cv2) with the DNN module
- NumPy
   
