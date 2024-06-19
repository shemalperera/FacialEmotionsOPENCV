# 🚀 Real-Time Face Emotion Recognition and Tracking 🎥

## Introduction
This project combines Deep Learning and Computer Vision to create an advanced real-time face emotion recognition and tracking system. This project is designed to detect faces, recognize their emotions, and track multiple individuals simultaneously within a video stream or real-time camera feed.

## Development Ladder
- Initial Model Development: Starting off with a CNN model designed to recognize emotions in static images. This initial model was trained on FER 2013 dataset of facial images categorized by emotion. <br>
- Below shows the Model architecture of Emotion Recognition of static faces. <br>

<p align="center">
<img src = "Model Architecture.jpg">
</p>

- The model was trained for 100 epochs and achieved an accuracy of 66.5%. We are currently optimizing the model to improve its accuracy.
- The emotion recognition model was then integrated into a real-time system using Haar cascade Frontalface for face detection.
- Then, the emotion recognition model was further integrated with YOLOv8 for tracking multiple people in real-time. This enabled the system to detect, recognize emotions, and track different individuals within the same frame. 

<p align="center">
<img src = "HaarCascade.png"> <br>
  https://www.researchgate.net/figure/Cascade-structure-for-Haar-classifiers_fig9_277929875
</p>
