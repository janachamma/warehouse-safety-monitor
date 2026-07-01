"""
Warehouse Safety Monitor
=========================
Author: Jana Chamma
Description: A real-time AI-powered warehouse safety monitoring system that:
  - Detects people and objects using YOLOv5
  - Counts persons and flags unauthorized devices
  - Generates safety alerts for dangerous scenes
  - Extracts text via OCR (signs, labels, warnings)
  - Logs all events with timestamps to JSON
  - Saves annotated snapshots as evidence

Inspired by real-world edge AI deployment work on robotic perception systems.
"""

import cv2
import torch
import easyocr
import numpy as np
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "yolo_model": "yolov5s",
    "confidence_threshold": 0.45,
    "iou_threshold": 0.45,
    "ocr_languages": ["en"],
    "ocr_gpu": False,
    "output_dir": "outputs",
    "font_scale": 0.6,
    "box_thickness": 2,
    "max_frames": 150,
    "snapshot_interval": 30,
}

# Safety rules
DANGER_OBJECTS = ["knife", "scissors", "fork"]
UNAUTHORIZED_DEVICES = ["cell phone", "laptop", "tv"]
PERSON_CLASS = "person"

# Alert levels
ALERT_LEVELS = {
    "CLEAR": ("Zone Clear", (0, 255, 0)),
    "WARNING": ("Warning — Person Detected", (0, 165, 255)),
    "DANGER": ("DANGER — Hazard Detected!", (0, 0, 255)),
    "UNAUTHORIZED": ("Alert — Unauthorized Device!", (0, 0, 255)),
}

# Color palette
COLORS = {
    "person": (0, 165, 255),       # Orange
    "danger": (0, 0, 255),         # Red
    "unauthorized": (0, 0, 200),   # Dark red
    "normal": (51, 255, 87),       # Green
    "text": (0, 255, 200),         # Cyan for OCR
}


# =============================================================================
# SAFETY MONITOR CLASS
# =============================================================================

class WarehouseSafetyMonitor:
    def __init__(self):
        print("\n" + "="*60)
        print("  WAREHOUSE SAFETY MONITOR")
        print("  AI-Powered Perception System")
        print("  Author: Jana Chamma")
        print("  Stack: YOLOv5 + EasyOCR + OpenCV")
        print("="*60)

        Path(CONFIG["output_dir"]).mkdir(exist_ok=True)

        self._load_models()

        # Session tracking
        self.event_log = []
        self.session_stats = {
            "start_time": datetime.now().isoformat(),
            "frames_processed": 0,
            "total_persons_detected": 0,
            "total_alerts": 0,
            "danger_events": 0,
            "unauthorized_device_events": 0,
            "objects_seen": {},
            "snapshots_saved": [],
        }

    def _load_models(self):
        print("\n Loading YOLOv5...")
        self.yolo = torch.hub.load(
            'ultralytics/yolov5', CONFIG["yolo_model"],
            pretrained=True, verbose=False
        )
        self.yolo.conf = CONFIG["confidence_threshold"]
        self.yolo.iou = CONFIG["iou_threshold"]
        self.yolo.cpu()
        print("  YOLOv5s ready")

        print("\n Loading EasyOCR...")
        self.ocr = easyocr.Reader(CONFIG["ocr_languages"], gpu=CONFIG["ocr_gpu"])
        print("  EasyOCR ready (CPU)")

    # =========================================================================
    # DETECTION
    # =========================================================================

    def detect_objects(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo(rgb)
        detections = []

        for *box, conf, cls in results.xyxy[0].tolist():
            label = self.yolo.names[int(cls)]
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [int(b) for b in box],
                "class_id": int(cls)
            })
            self.session_stats["objects_seen"][label] = \
                self.session_stats["objects_seen"].get(label, 0) + 1

        return detections

    def extract_text(self, frame):
        results = self.ocr.readtext(frame)
        text_regions = []
        for (bbox, text, conf) in results:
            if conf > 0.4 and len(text.strip()) > 1:
                pts = np.array(bbox, dtype=np.int32)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                text_regions.append({
                    "text": text.strip(),
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        return text_regions

    # =========================================================================
    # SAFETY ANALYSIS
    # =========================================================================

    def analyze_safety(self, detections, text_regions):
        """Core safety logic — determine alert level and generate report."""
        labels = [d["label"] for d in detections]
        person_count = labels.count(PERSON_CLASS)
        dangers = [l for l in labels if l in DANGER_OBJECTS]
        unauthorized = [l for l in labels if l in UNAUTHORIZED_DEVICES]

        # Determine alert level
        if dangers:
            alert = "DANGER"
            self.session_stats["danger_events"] += 1
        elif unauthorized:
            alert = "UNAUTHORIZED"
            self.session_stats["unauthorized_device_events"] += 1
        elif person_count > 0:
            alert = "WARNING"
        else:
            alert = "CLEAR"

        if alert != "CLEAR":
            self.session_stats["total_alerts"] += 1

        self.session_stats["total_persons_detected"] += person_count

        # Build safety report
        report = {
            "timestamp": datetime.now().isoformat(),
            "alert_level": alert,
            "alert_message": ALERT_LEVELS[alert][0],
            "person_count": person_count,
            "danger_objects": dangers,
            "unauthorized_devices": unauthorized,
            "all_detections": [d["label"] for d in detections],
            "object_count": len(detections),
            "scene_density": self._get_density(len(detections)),
            "text_detected": [t["text"] for t in text_regions],
        }

        # Log event
        self.event_log.append(report)
        return report

    def _get_density(self, count):
        if count == 0: return "empty"
        elif count <= 3: return "sparse"
        elif count <= 8: return "moderate"
        else: return "dense"

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def draw_frame(self, frame, detections, text_regions, safety_report):
        output = frame.copy()
        h, w = frame.shape[:2]
        alert = safety_report["alert_level"]
        alert_msg, alert_color = ALERT_LEVELS[alert]

        # Draw alert banner at top
        cv2.rectangle(output, (0, 0), (w, 50), alert_color, -1)
        cv2.putText(output, f"  {alert_msg}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]

            # Choose color by category
            if label == PERSON_CLASS:
                color = COLORS["person"]
            elif label in DANGER_OBJECTS:
                color = COLORS["danger"]
            elif label in UNAUTHORIZED_DEVICES:
                color = COLORS["unauthorized"]
            else:
                color = COLORS["normal"]

            cv2.rectangle(output, (x1, y1), (x2, y2), color, CONFIG["box_thickness"])

            display = f"{label} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX,
                                           CONFIG["font_scale"], 1)
            cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(output, display, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, CONFIG["font_scale"],
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Draw OCR regions
        for txt in text_regions:
            x1, y1, x2, y2 = txt["bbox"]
            cv2.rectangle(output, (x1, y1), (x2, y2), COLORS["text"], 2)
            cv2.putText(output, f'OCR: "{txt["text"]}"', (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLORS["text"], 1, cv2.LINE_AA)

        # HUD panel (bottom)
        hud_bg_y = h - 90
        cv2.rectangle(output, (0, hud_bg_y), (w, h), (20, 20, 20), -1)

        hud_lines = [
            f"Persons: {safety_report['person_count']}  |  "
            f"Objects: {safety_report['object_count']}  |  "
            f"Scene: {safety_report['scene_density']}",
            f"Dangers: {safety_report['danger_objects'] or 'None'}  |  "
            f"Unauthorized: {safety_report['unauthorized_devices'] or 'None'}",
            f"OCR Text: {safety_report['text_detected'][:2] or 'None'}  |  "
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
        ]

        for i, line in enumerate(hud_lines):
            cv2.putText(output, line, (10, hud_bg_y + 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (200, 200, 200), 1, cv2.LINE_AA)

        return output

    # =========================================================================
    # WEBCAM MODE
    # =========================================================================

    def process_webcam(self):
        print("\n Starting webcam safety monitoring...")
        print(f" Saving snapshots every {CONFIG['snapshot_interval']} frames")
        print(f" Total frames: {CONFIG['max_frames']} (~2-3 mins on CPU)")
        print(" Point webcam at yourself, objects, or text\n")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam")
            return

        frame_count = 0
        ocr_done = False
        last_text = []

        while frame_count < CONFIG["max_frames"]:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            self.session_stats["frames_processed"] += 1

            # Object detection every frame
            detections = self.detect_objects(frame)

            # OCR once at frame 5
            if frame_count == 5 and not ocr_done:
                print("  Running OCR scan...")
                last_text = self.extract_text(frame)
                ocr_done = True
                print(f"  Found {len(last_text)} text regions")

            # Safety analysis
            safety = self.analyze_safety(detections, last_text)

            # Draw annotated frame
            output = self.draw_frame(frame, detections, last_text, safety)

            # Save snapshot every N frames
            if frame_count % CONFIG["snapshot_interval"] == 0:
                snap_path = f"{CONFIG['output_dir']}/safety_snap_{frame_count:04d}.jpg"
                cv2.imwrite(snap_path, output)
                self.session_stats["snapshots_saved"].append(snap_path)
                print(f"  [{frame_count}/{CONFIG['max_frames']}] "
                      f"Snapshot saved | Alert: {safety['alert_level']} | "
                      f"Persons: {safety['person_count']} | "
                      f"Objects: {safety['object_count']}")

        cap.release()
        self._save_final_report()
        self._print_summary()

    # =========================================================================
    # IMAGE MODE
    # =========================================================================

    def process_image(self, image_path: str):
        print(f"\n Processing: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not load: {image_path}")
            return

        t0 = time.time()
        detections = self.detect_objects(frame)
        text_regions = self.extract_text(frame)
        safety = self.analyze_safety(detections, text_regions)
        elapsed = round(time.time() - t0, 2)

        output = self.draw_frame(frame, detections, text_regions, safety)

        out_path = f"{CONFIG['output_dir']}/safety_{Path(image_path).stem}.jpg"
        cv2.imwrite(out_path, output)

        report_path = f"{CONFIG['output_dir']}/report_{Path(image_path).stem}.json"
        with open(report_path, "w") as f:
            json.dump({**safety, "processing_time": elapsed,
                       "input": image_path}, f, indent=2)

        print(f"\n  Alert Level:  {safety['alert_level']}")
        print(f"  Persons:      {safety['person_count']}")
        print(f"  Objects:      {safety['object_count']}")
        print(f"  Dangers:      {safety['danger_objects'] or 'None'}")
        print(f"  Unauthorized: {safety['unauthorized_devices'] or 'None'}")
        print(f"  OCR Text:     {safety['text_detected'] or 'None'}")
        print(f"  Time:         {elapsed}s")
        print(f"\n  Saved: {out_path}")
        print(f"  Report: {report_path}")

    # =========================================================================
    # REPORTING
    # =========================================================================

    def _save_final_report(self):
        report = {
            "session_summary": self.session_stats,
            "event_log": self.event_log[-20:],  # last 20 events
        }
        path = f"{CONFIG['output_dir']}/safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Full report saved: {path}")

    def _print_summary(self):
        s = self.session_stats
        print(f"\n{'='*55}")
        print("  SAFETY SESSION SUMMARY")
        print(f"{'='*55}")
        print(f"  Frames processed:        {s['frames_processed']}")
        print(f"  Total alerts triggered:  {s['total_alerts']}")
        print(f"  Persons detected:        {s['total_persons_detected']}")
        print(f"  Danger events:           {s['danger_events']}")
        print(f"  Unauthorized devices:    {s['unauthorized_device_events']}")
        print(f"  Snapshots saved:         {len(s['snapshots_saved'])}")
        if s["objects_seen"]:
            print(f"\n  Objects seen:")
            for obj, count in sorted(s["objects_seen"].items(),
                                      key=lambda x: x[1], reverse=True)[:8]:
                print(f"    - {obj}: {count}x")
        print(f"{'='*55}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Warehouse Safety Monitor — YOLOv5 + EasyOCR"
    )
    parser.add_argument("--mode", choices=["image", "webcam"],
                        default="webcam")
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    monitor = WarehouseSafetyMonitor()

    if args.mode == "image":
        if not args.input:
            print("Provide --input path. Example:")
            print("  python pipeline.py --mode image --input photo.jpg")
            return
        monitor.process_image(args.input)
    else:
        monitor.process_webcam()


if __name__ == "__main__":
    main()
