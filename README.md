# Condition-Monitoring-Project


This project provides a full toolchain for measuring and analyzing vibrations from a domestic fan using an MPU6050 sensor and an Arduino Uno. The aim is to detect mechanical faults such as unbalance using frequency-domain analysis (FFT).

---

## 📌 Project Overview

Using an MPU6050 3-axis accelerometer/gyroscope module, the system acquires real-time vibration data via an Arduino Uno and analyzes it with Python scripts. It allows:
- 10-second vibration data acquisition
- Uniform time interpolation
- Frequency-domain analysis (FFT)
- Automatic detection of unbalance on the Y-axis
- Exporting raw `aY` values to `.txt` files
- Plotting acceleration and spectral data

---

## 🛠️ Hardware Requirements

- Arduino Uno
- MPU6050 sensor (connected via I2C: VCC, GND, SDA, SCL)
- USB cable for serial connection to PC

---

## 🧑‍💻 Code Structure

### 🔹 `fan_analysis.py`
> Main Python script for data acquisition, FFT, and unbalance detection.

### 🔹 `UnoCode.ino`
> Arduino sketch to be uploaded to the Uno. It sends acceleration data (`gX,gY,gZ`) over serial.

### 🔹 *(Coming soon)* `analyzer_gui.py`
> A Python-based GUI tool to load and visualize previously recorded raw data files.

---

## 🚀 How to Use

1. Upload `UnoCode.ino` to your Arduino Uno using the Arduino IDE.
2. Connect the MPU6050 sensor to the Uno via I2C.
3. Run `fan_analysis.py` on your PC (requires `pyserial`, `numpy`, `scipy`, `matplotlib`).
4. Follow on-screen instructions. After 10 seconds of data capture, the script will:
   - Save `aY` to a `.txt` file in your local folder
   - Display time-domain and FFT plots
   - Print diagnostic messages about unbalance

---

## 🎓 Presentation Resources

- 📑 [Download the PowerPoint presentation: `ProjectPresentation.pptx`](https://your_link_here)
- 📺 [Watch the presentation on YouTube](https://your_youtube_video_link_here)

---

## 📂 Output Files

- `acquisition_YYYY-MM-DD_HHMM.txt`: raw `aY` acceleration values
- Diagnostic messages on terminal (e.g., `"⚠️ Warning: amplitude threshold exceeded at 12.5 Hz"`)
- Time series and FFT plots for all three axes

---

## 📄 License

This project is released for academic and educational use only.

---

## 🤝 Acknowledgements

Special thanks to the **MEMM1263 Condition Monitoring** course and its instructors for the theoretical foundations and guidance.
