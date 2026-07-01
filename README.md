# 🏭 Warehouse Safety Monitor

> An AI-powered real-time safety monitoring system using YOLOv5 object detection and EasyOCR — detecting persons, hazards, and unauthorized devices with live alert classification and structured JSON logging.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-red)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Text%20Extraction-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This system mimics a real-world AI perception pipeline deployed on edge devices for industrial safety monitoring. It combines object detection, OCR, and a rule-based safety engine to classify scenes and trigger alerts in real time.

```
Webcam / Image Input
        │
        ▼
┌──────────────────┐
│  YOLOv5 Detector │  ← Detects persons, devices, hazards (80+ classes)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  EasyOCR Scanner │  ← Reads warning signs, labels, text in scene
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Safety Engine   │  ← Classifies alert level based on detection rules
└────────┬─────────┘
         │
         ▼
Annotated Frame + JSON Safety Report
```

---

## 🚨 Alert System

| Alert Level | Trigger | Banner Color |
|---|---|---|
| ✅ CLEAR | No persons or hazards | Green |
| ⚠️ WARNING | Person detected in zone | Orange |
| 🔴 DANGER | Hazardous object detected (knife, scissors) | Red |
| 🔴 UNAUTHORIZED | Unauthorized device detected (phone, laptop) | Red |

---

## 🛠️ Features

- **Person Detection & Counting** — tracks how many people are in the monitored zone
- **Hazard Detection** — flags dangerous objects (knives, scissors, forks)
- **Unauthorized Device Alerts** — detects phones and laptops in restricted areas
- **OCR Text Reading** — reads warning signs, labels, and text visible in the scene
- **Color-coded Bounding Boxes** — green for safe objects, orange for persons, red for threats
- **Live HUD Overlay** — real-time stats panel showing counts, alerts, and timestamp
- **JSON Event Logging** — every frame's safety status logged with timestamp
- **Session Summary Report** — full safety report saved at end of session
- **CPU Optimized** — runs on standard hardware without GPU

---

## 📊 Sample JSON Safety Report

```json
{
  "timestamp": "2026-06-28T14:32:11",
  "alert_level": "UNAUTHORIZED",
  "alert_message": "Alert — Unauthorized Device!",
  "person_count": 1,
  "danger_objects": [],
  "unauthorized_devices": ["cell phone"],
  "all_detections": ["person", "cell phone", "chair"],
  "object_count": 3,
  "scene_density": "sparse",
  "text_detected": ["CAUTION", "ZONE B"]
}
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/janachamma/warehouse-safety-monitor.git
cd warehouse-safety-monitor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run

**Webcam mode (live monitoring):**
```bash
python pipeline.py --mode webcam
```

**Image mode (single image analysis):**
```bash
python pipeline.py --mode image --input photo.jpg
```

---

## 📸 Demo Scenarios

| Scenario | What to show | Expected Alert |
|---|---|---|
| Empty zone | Empty desk/room | ✅ CLEAR |
| Person enters | Sit in front of webcam | ⚠️ WARNING |
| Unauthorized device | Hold your phone | 🔴 UNAUTHORIZED |
| Hazard | Hold scissors | 🔴 DANGER |
| Text reading | Hold paper with text | OCR extracts text |

---

## 📁 Project Structure

```
warehouse-safety-monitor/
├── pipeline.py                  # Main safety monitor
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── outputs/
    ├── safety_snap_XXXX.jpg     # Annotated snapshots
    ├── report_*.json            # Per-image safety reports
    └── safety_report_*.json     # Full session summary
```

---

## 🔧 Configuration

```python
CONFIG = {
    "yolo_model": "yolov5s",        # yolov5s / yolov5m / yolov5l
    "confidence_threshold": 0.45,
    "ocr_languages": ["en"],        # Add "ar" for Arabic
    "ocr_gpu": False,               # Set True if GPU available
    "max_frames": 150,              # Session length
    "snapshot_interval": 30,        # Save every N frames
}

DANGER_OBJECTS = ["knife", "scissors", "fork"]
UNAUTHORIZED_DEVICES = ["cell phone", "laptop", "tv"]
```

---

## 💡 Key Concepts Demonstrated

- **Real-time object detection** — YOLOv5 inference on live webcam feed
- **Rule-based safety engine** — alert classification from detection results
- **Multi-modal perception** — visual detection + OCR text reading combined
- **Structured event logging** — timestamped JSON logs per frame
- **CPU inference optimization** — OCR throttled to reduce latency
- **Production-style architecture** — config-driven, modular class design
- **Edge AI patterns** — designed for deployment on constrained hardware

---

## 👩‍💻 Author

**Jana Chamma**  
AI/ML Engineer | Computer Vision | Edge AI  
[LinkedIn](https://www.linkedin.com/in/jana-chamma-26b7212b3/) | [Portfolio](https://janachamma.github.io/)

---

## 📄 License

MIT License
