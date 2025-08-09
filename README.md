# Facial Landmark Detection Using PyTorch

A real-time facial landmark detection system using PyTorch and HRNetV2.

## Features
- Real-time facial landmark detection using webcam
- PyTorch-based implementation using HRNetV2
- Visual display of detected landmarks

## Installation
1. Clone this repository
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Download the HRNetV2 model and place it in the models directory

## Usage
Run the main script:
```bash
python src/main.py
```

## Project Structure
```
├── models/              # Model weights and configurations
├── src/                 # Source code
│   ├── landmark_detector.py    # Facial landmark detection implementation
│   └── main.py                # Main application entry point
└── utils/               # Utility functions
    └── video_utils.py          # Video and webcam handling utilities
```
