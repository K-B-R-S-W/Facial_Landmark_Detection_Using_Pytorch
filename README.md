# 🎯 Facial Landmark Detection Using PyTorch

A **real-time** facial landmark detection system powered by **PyTorch** and the **HRNetV2** architecture.  
It supports **webcam-based detection**, **pretrained model loading**, and is **easily extensible** for research or production use.

---

## 🚀 Features
- 📸 **Real-time facial landmark detection** from webcam  
- 🧠 **PyTorch-based HRNetV2** backbone with landmark detection head  
- 🎨 Visual display & saving of detected landmarks  
- 🛠 Modular code for **easy extension & debugging**  
- ✅ Includes **test script** for model and weight verification  

---

## 📦 Installation

1. **Clone this repository**
    ```bash
    git clone https://github.com/K-B-R-S-W/Facial_Landmark_Detection_Using_Pytorch.git
    cd Facial_Landmark_Detection_Using_Pytorch
    ```

2. **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the HRNetV2 pretrained weights**
    - Download: `hrnetv2_w32_imagenet_pretrained.pth`  
    - Get official weights from: [HRNet GitHub](https://github.com/HRNet/HRNet-Image-Classification)
    - After You trained the model, Place it in the `models/` directory.  

---

## 🖥 Usage

### ▶ Run the Main Application  
Start webcam-based real-time detection:
```bash
python main.py
```

### 🧪 Run the Model Test Script  
Verify model architecture, weight loading & detector functionality:
```bash
python test_model.py
```

---

## 📂 Project Structure
```
Facial_Landmark_Detection_Using_Pytorch/
├── main.py                       
├── test_model.py                 
├── requirements.txt
├── models/
│   └── hrnetv2_w32_imagenet_pretrained.pth  
├── src/
│   ├── __init__.py
│   └── landmark_detector.py   
├── utils/
│   ├── __init__.py
│   └── video_utils.py        
└── README.md
```

---

## 📝 Notes
- ⚠ **Pretrained model (you have to train your own model using the Pretrained Model) file not included**. 
- 📂 You can **train your own model** and place it in the `models/` folder.  
- 🔄 The code is modular — extend it for **other datasets** or **detection tasks**.  
- 🛠 Use `test_model.py` for troubleshooting & debugging.  

---

## 📚 References
- [📄 HRNet: High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514)  
- [💻 HRNet Official GitHub](https://github.com/HRNet/HRNet-Image-Classification)  

---

## 📮 Support

**📧 Email:** [k.b.ravindusankalpaac@gmail.com](mailto:k.b.ravindusankalpaac@gmail.com)  
**🐞 Bug Reports:** [GitHub Issues](https://github.com/K-B-R-S-W/Facial_Landmark_Detection_Using_Pytorch/issues)  
**📚 Documentation:** [Project Wiki](https://github.com/K-B-R-S-W/Facial_Landmark_Detection_Using_Pytorch/wiki)  
**💭 Discussions:** [GitHub Discussions](https://github.com/K-B-R-S-W/Facial_Landmark_Detection_Using_Pytorch/discussions)  

---

## ⭐ Support This Project
If you find this project helpful, please give it a **⭐ star** on GitHub — it motivates me to keep improving! 🚀

