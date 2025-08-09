
# Facial Landmark Detection Using PyTorch

This project provides a real-time facial landmark detection system using PyTorch and the HRNetV2 architecture. It supports webcam-based detection, pretrained model loading, and easy extensibility for research or application use.

## Features
- Real-time facial landmark detection from webcam
- PyTorch-based HRNetV2 backbone with landmark detection head
- Visual display and saving of detected landmarks
- Modular code for easy extension and debugging
- Includes test script for model and weight verification

## Installation
1. **Clone this repository**
2. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Download the HRNetV2 pretrained weights:**
    - Download `hrnetv2_w32_imagenet_pretrained.pth` and place it in the `models/` directory. See [HRNet GitHub](https://github.com/HRNet/HRNet-Image-Classification) for official weights.

## Usage

### Run the Main Application
This will start the webcam and display real-time facial landmark detection:
```bash
python main.py
```

### Run the Model Test Script
To verify model architecture, weight loading, and detector functionality:
```bash
python test_model.py
```

## Project Structure
```
Facial_Landmark_Detection_Using_Pytorch/
├── main.py                # Main application (webcam landmark detection)
├── test_model.py          # Script to test model loading and inference
├── requirements.txt       # Python dependencies
├── models/
│   └── hrnetv2_w32_imagenet_pretrained.pth  # Pretrained HRNetV2 weights
├── src/
│   ├── __init__.py
│   └── landmark_detector.py   # HRNetV2 and landmark detection implementation
├── utils/
│   ├── __init__.py
│   └── video_utils.py         # Video and webcam utilities
└── README.md
```

## Notes
- The pretrained model file is **not included** due to size. Download it manually and place it in the `models/` directory.
- The code is modular and can be extended for other landmark datasets or detection tasks.
- For troubleshooting, use `test_model.py` to debug model loading and inference.

## References
- [HRNet: High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514)
- [HRNet Official GitHub](https://github.com/HRNet/HRNet-Image-Classification)

---

## 📮 Support

**📧 Email:** [k.b.ravindusankalpaac@gmail.com](mailto:k.b.ravindusankalpaac@gmail.com)  
**🐞 Bug Reports:** [GitHub Issues](https://github.com/K-B-R-S-W/Realtime_Sri_Lankan_License_Plate_Capture_System_Using_Yolo_V11/issues)  
**📚 Documentation:** See the project [Wiki](https://github.com/K-B-R-S-W/Realtime_Sri_Lankan_License_Plate_Capture_System_Using_Yolo_V11/wiki)  
**💭 Discussions:** Join the [GitHub Discussions](https://github.com/K-B-R-S-W/Realtime_Sri_Lankan_License_Plate_Capture_System_Using_Yolo_V11/discussions)

---

## ⭐ Support This Project

If you find this project helpful, please consider giving it a **⭐ star** on GitHub!