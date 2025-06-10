以下は、**そのままGitHubの`README.md`に貼り付けて使える**Markdown形式の全文です。`Author`セクションは省いてあります。

---

```markdown
# Human Distance and Position Estimation using YOLOv8 and MiDaS

This project uses **YOLOv8** for real-time human detection and **MiDaS** for monocular depth estimation. It estimates the **3D position (X, Y, Z)** of each detected person in the camera coordinate system from a single image captured by a webcam or IP camera.

---

## 📌 Overview

- **YOLOv8 (Ultralytics)** is used to detect persons in an input image.
- **MiDaS (Intel-ISL)** is used to estimate depth information from a single RGB image.
- Based on the camera intrinsics and depth map, the program computes the **3D location (X, Y, Z)** of detected persons.
- Visualization includes bounding boxes and real-world coordinates overlayed on the image.

---

## 🎯 Purpose

The main goal is to build a lightweight 3D human localization system using only a **single camera**, without LiDAR or stereo cameras. This can be useful in applications such as:

- People tracking in indoor environments
- Human-robot interaction
- Privacy-preserving monitoring systems
- Drone or surveillance camera systems

---

## 🖥️ Example Output

- Detected persons with bounding boxes
- Estimated real-world coordinates printed on screen and shown in the image:
  
```

XYZ: (X.XX, Y.YY, Z.ZZ) m

````

![example output](./example_output.jpg)  <!-- Replace with actual result image -->

---

## 🧱 System Requirements

- Python 3.8+
- OpenCV (`cv2`)
- PyTorch
- NumPy
- Ultralytics YOLOv8
- Intel MiDaS (loaded via `torch.hub`)

---

## ⚙️ Installation

### 1. Clone the repository (if applicable)
```bash
git clone https://github.com/yourusername/yolo-midas-3d-localization.git
cd yolo-midas-3d-localization
````

### 2. Create and activate a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows
```

### 3. Install required packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy ultralytics
```

---

## 🚀 Run the Program

Edit the image path in the script:

```python
image_path = "/path/to/your/image.jpg"
```

Then run:

```bash
python yolo_midas_xyz.py
```

You will see the detected persons, their bounding boxes, and their 3D positions shown on the image.

---

## 📷 Camera Parameters

The current implementation assumes:

```python
fx = 500.0
fy = 500.0
cx = image_width / 2
cy = image_height / 2
```

> These are **approximations**. For accurate localization, use your camera's intrinsic calibration.

---

## 📄 License

This repository uses components under the following licenses:

* [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) - GPL-3.0
* [MiDaS by Intel ISL](https://github.com/intel-isl/MiDaS) - MIT

Your own code may be distributed under MIT or another compatible license.

```

---

必要があれば、**Webカメラ対応のコード例**や**日本語版README**も作成しますので、お気軽にどうぞ。
```
