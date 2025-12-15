# Advanced Measurement Tool

A **professional-grade real-time object detection, tracking, and measurement system** built with **Python and OpenCV**. This tool detects physical objects from a live camera feed, filters them by real-world size (cm²), classifies shapes, smooths measurements, tracks objects across frames, and provides a polished interactive UI.

The project is designed for **accuracy, performance, and stability**, with special attention to **FPS optimization**, **noise rejection**, and **long-session reliability**.

---

## ✨ Key Features

* 📷 **Real-time webcam-based object detection**
* 📐 **Real-world measurement in centimeters (cm, cm²)**
* 🎯 **Smart size filtering** (Tiny → Very Large)
* 🔄 **Object tracking across frames** (centroid-based)
* 🧮 **Accurate geometry calculations** (area, perimeter, width, height)
* 🧠 **Shape classification** (Circle, Rectangle, Polygon, etc.)
* 📊 **Confidence scoring system**
* 📉 **Measurement smoothing (EMA)** to reduce jitter
* 🖱️ **Clickable OpenCV UI (mouse-based)**
* ⚡ **Optimized for high FPS & stability**
* 💾 **JSON export of measurements & statistics**

---

## 🧩 Size Categories (Real-World)

| Category   | Area Range (cm²) |
| ---------- | ---------------- |
| Tiny       | < 5              |
| Small      | 5 – 25           |
| Medium     | 25 – 100         |
| Large      | 100 – 300        |
| Very Large | > 300            |

Each category can be **enabled or disabled independently** using the FILTER menu.

---

## 🖥️ System Requirements

### Software

* Python **3.8 or higher**
* OpenCV (`opencv-python`)
* NumPy
* Tkinter (optional, for DPI detection)

Install dependencies:

```bash
pip install opencv-python numpy
```

### Hardware

* Webcam (720p or higher recommended)
* Standard laptop/desktop CPU (no GPU required)

---

## 🚀 How to Run

```bash
python main.py
```

On launch, the system will:

* Detect monitor DPI and resolution
* Detect camera resolution and FPS
* Automatically calibrate **pixels-per-centimeter**

---

## 🕹️ User Interface Controls

### Main Buttons

| Button  | Function                               |
| ------- | -------------------------------------- |
| DETECT  | Run object detection once              |
| AUTO    | Enable continuous detection (FPS-safe) |
| ANALYZE | Save measurement of selected object    |
| FILTER  | Open size category selector            |
| STATS   | Print statistics to console            |
| CLEAR   | Clear detections & smoothing           |
| SAVE    | Export measurements to JSON            |
| EXIT    | Close application                      |

### Mouse Interaction

* **Left Click** → Select buttons
* Detected objects are automatically indexed

---

## 📊 Measurement Output

Each measured object includes:

* Width (cm)
* Height (cm)
* Area (cm²)
* Perimeter (cm)
* Shape classification
* Size category
* Vertex count
* Confidence score (%)
* Timestamp

Measurements are smoothed using an **Exponential Moving Average (EMA)** to reduce frame noise.

---

## 📈 Performance & FPS Optimization

The system is optimized to prevent FPS drops by:

* Caching kernels and CLAHE objects
* Avoiding redundant detections
* Tracking objects between detection cycles
* Limiting full detection frequency in AUTO mode
* Dropping frames safely if processing lags

Target performance:

* **25–30 FPS** on standard laptops
* Stable over long runtime sessions

---

## 💾 Data Saving Format

Measurements are saved as timestamped JSON files:

```json
{
  "system": {
    "monitor": "1920x1080@96DPI",
    "camera": "1280x720@30FPS",
    "pixels_per_cm": 37.79
  },
  "measurements": [...],
  "total": 12,
  "size_filters_used": ["Medium", "Large"],
  "statistics": {
    "total_objects": 12
  }
}
```

---

## 🧠 Architecture Overview

* **ObjectTracker**

  * Maintains persistent IDs across frames
  * Prevents flickering and duplicate detection

* **SizeFilterManager**

  * Converts cm² → px² using calibration
  * Controls detection range precisely

* **MeasurementTool**

  * Detection pipeline
  * UI rendering
  * Measurement logic
  * Data persistence

The system follows a **modular, OOP-based design** for maintainability and scalability.

---

## 🛡️ Stability & Error Handling

* Guards against:

  * Division by zero
  * Empty contours
  * Camera read failures
* Graceful exits
* Safe resource release (camera & windows)

---

## 🔧 Common Tips for Best Accuracy

* Use a well-lit environment
* Avoid reflective surfaces
* Keep camera stable
* Place objects flat and clearly separated
* Use consistent camera distance

---

## 📌 Limitations

* Calibration is DPI-based (screen-dependent)
* Absolute accuracy depends on camera alignment
* Not intended for sub-millimeter precision

---

## 🔮 Future Improvements (Optional)

* Reference-object calibration (credit card / ruler)
* CSV export
* Video recording
* Mobile / APK optimization
* PyInstaller executable build

---

## 📜 License

This project is intended for **educational and portfolio use**. You may modify and extend it freely.

---

## ✅ Status

✔ Fully functional
✔ FPS-optimized
✔ Stable for long sessions
✔ Production-quality OpenCV project

---

**Advanced Measurement Tool v8.1** – Built for precision, speed, and reliability.
