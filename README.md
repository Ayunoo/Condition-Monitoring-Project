# 🌀 Fan Vibration Monitoring & Fault Detection

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

### 🔹 `CollectingData.py`
> Python script for data acquisition, FFT computation, and automatic unbalance detection.

### 🔹 `UnoCode.ino`
> Arduino sketch to be uploaded to the Uno. It sends acceleration data (`gX,gY,gZ`) over serial.

### 🔹 `AnalyzingData.py` *(GUI version)*
> A Python-based graphical interface to load and visualize previously recorded raw data files (such as exported `.txt` files).

---

## 🚀 How to Use

1. Upload `UnoCode.ino` to your Arduino Uno using the Arduino IDE.
2. Connect the MPU6050 sensor to the Uno via I2C.
3. Run `AnalyzingData.py` on your PC (requires `pyserial`, `numpy`, `scipy`, `matplotlib`).
4. Follow on-screen instructions. After 10 seconds of data capture, the script will:
   - Save `aY` to a `.txt` file in your local folder
   - Display time-domain and FFT plots
   - Print diagnostic messages about unbalance

Alternatively, use the GUI version of `AnalyzingData.py` to load and analyze previously acquired `.txt` files.

---

## 🎓 Presentation Resources

- 📑 [Download the PowerPoint presentation: `ProjectPresentation.pptx`](https://github.com/Ayunoo/Condition-Monitoring-Project/blob/main/PresentationProject.pptx)
- 📺 [Watch the project presentation on YouTube](https://youtu.be/3-jeOoLR8ko)

---

## 📂 Output Files

- `acquisition_YYYY-MM-DD_HHMM.txt`: raw `aY` acceleration values  
- Diagnostic messages in the terminal (e.g., `"⚠️ Warning: amplitude threshold exceeded at 12.5 Hz"`)  
- Time series and FFT plots for all three axes  

---

## 📄 License

This project is released for academic and educational use only.

---

## 🤝 Acknowledgements

Special thanks to:
- The **MEMM1263 Condition Monitoring** course and Dr Zair Asrar for the theoretical framework.  
- The **Electrical Laboratory** for providing access to the equipment and space used during data collection.
