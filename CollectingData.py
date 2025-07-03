"""
Fan Vibration Acquisition and Analysis Tool

This script connects to an Arduino Uno via serial port to acquire 3-axis acceleration
data from an MPU6050 sensor placed on a rotating fan. The goal is to monitor vibration
signals and analyze their frequency content (FFT) in order to detect potential unbalance faults.

The script performs:
- 10-second real-time data acquisition on X, Y, Z axes
- Interpolation for uniform time sampling
- FFT computation to extract dominant frequencies
- Automatic detection of excessive amplitude on the Y axis (excluding known fixed-frequency noise near 22 Hz)
- Saves aY data to a .txt file
- Plots acceleration signals and their frequency spectrum

Author: Adrien Pierre MALCOIFFE & Wei YAGE
Date: 03/07/2025
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from datetime import datetime

# === CONFIGURATION ===
PORT = 'COM9'  # Adapt to the USB port where the Arduino Uno is connected
BAUD = 115200
DURATION = 10  # seconds
SAVE_DIR = r"C:\Users\Adrien"  # Adapt to your machine

# === Prepare file name for aY recording ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
file_name = f"acquisition_{timestamp}.txt"
file_path = os.path.join(SAVE_DIR, file_name)

# === Serial connection ===
ser = serial.Serial(PORT, BAUD)
time.sleep(2)

timestamps = []
accX, accY, accZ = [], [], []

start_time = time.time()

print("Acquisition in progress...")

while time.time() - start_time < DURATION:
    try:
        line = ser.readline().decode().strip()
        parts = line.split(",")
        if len(parts) != 3:
            continue

        gX = float(parts[0])
        gY = float(parts[1])
        gZ = float(parts[2])
        aX = gX * 9.81
        aY = gY * 9.81
        aZ = gZ * 9.81

        now = time.time()
        timestamps.append(now - start_time)
        accX.append(aX)
        accY.append(aY)
        accZ.append(aZ)

    except ValueError:
        continue

ser.close()
print("Acquisition completed.")

# === Save aY to .txt file ===
try:
    with open(file_path, 'w') as f:
        for ay_val in accY:
            f.write(f"{ay_val}\n")
    print(f"aY values saved to: {file_path}")
except Exception as e:
    print(f"File saving error: {e}")

# === Uniform interpolation ===
t = np.array(timestamps)
fs_est = len(t) / (t[-1] - t[0])
t_uniform = np.linspace(t[0], t[-1], len(t))

def interpolate_signal(t_raw, signal):
    return interp1d(t_raw, signal, kind='linear')(t_uniform)

aX = interpolate_signal(t, accX)
aY = interpolate_signal(t, accY)
aZ = interpolate_signal(t, accZ)

# === FFT computation ===
N = len(t_uniform)
freqs = np.fft.fftfreq(N, d=1/fs_est)
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]

def compute_fft(a):
    fft_vals = np.fft.fft(a)
    amps = 2 * np.abs(fft_vals[pos_mask]) / N
    f_dom = freqs_pos[np.argmax(amps)]
    return amps, f_dom

ampX, fX = compute_fft(aX)
ampY, fY = compute_fft(aY)
ampZ, fZ = compute_fft(aZ)

print(f"Dominant frequency X: {fX:.2f} Hz")
print(f"Dominant frequency Y: {fY:.2f} Hz")
print(f"Dominant frequency Z: {fZ:.2f} Hz")

# === Automatic unbalance detection on Y-axis (ignoring 22 Hz region) ===
threshold = 0.2  # m/s²
ignore_range = (21.5, 22.5)  # Hz

# Exclude 22 Hz region
filtered_indices = (freqs_pos < ignore_range[0]) | (freqs_pos > ignore_range[1])
filtered_amps = ampY[filtered_indices]
filtered_freqs = freqs_pos[filtered_indices]

# Check if any other peak exceeds the threshold
if len(filtered_amps) > 0:
    max_amp = np.max(filtered_amps)
    dom_freq = filtered_freqs[np.argmax(filtered_amps)]

    if max_amp > threshold:
        print(f"⚠️ Warning: amplitude threshold of {threshold} exceeded. "
              f"Detected amplitude = {max_amp:.2f} at {dom_freq:.2f} Hz")
    else:
        print("✅ Normal behaviour: no unbalance detected.")

# === PLOTTING ===
fig, axs = plt.subplots(3, 2, figsize=(14, 8))

# a(t)
axs[0, 0].plot(t_uniform, aX, label='aX')
axs[1, 0].plot(t_uniform, aY, label='aY', color='orange')
axs[2, 0].plot(t_uniform, aZ, label='aZ', color='green')

# FFT
axs[0, 1].plot(freqs_pos, ampX)
axs[0, 1].axvline(fX, color='red', linestyle='--', label=f'{fX:.2f} Hz')
axs[1, 1].plot(freqs_pos, ampY, color='orange')
axs[1, 1].axvline(fY, color='red', linestyle='--', label=f'{fY:.2f} Hz')
axs[2, 1].plot(freqs_pos, ampZ, color='green')
axs[2, 1].axvline(fZ, color='red', linestyle='--', label=f'{fZ:.2f} Hz')

# Formatting
for i, label in enumerate(['X', 'Y', 'Z']):
    axs[i, 0].set_title(f'a{label}(t)')
    axs[i, 0].set_ylabel('Acceleration (m/s²)')
    axs[i, 0].grid()
    axs[i, 0].legend()

    axs[i, 1].set_title(f'FFT a{label}')
    axs[i, 1].set_xlabel('Frequency (Hz)')
    axs[i, 1].set_ylabel('Amplitude')
    axs[i, 1].grid()
    axs[i, 1].legend()

plt.tight_layout()
plt.show()
