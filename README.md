## Athena Audio Signal Processing Suite 🔊

This project is a comprehensive solution for multi-speaker audio signal processing, leveraging advanced techniques from signal processing and deep learning. The primary goal is to accurately separate, denoise, and classify individual speakers from a mixed audio source, emulating a real-world "cocktail party" scenario.

-----

### 🚀 Getting Started

To run this project, follow these simple steps:

1.  **Audio Files:** Replace the placeholder files `sample_male.wav` and `sample_female.wav` in the project directory with your desired audio samples.
2.  **Dependencies:** Install all required libraries by running the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execution:** Run the main script to process your audio files:
    ```bash
    python main.py
    ```
4.  **GPU Support:** The `predict.py` script is optimized for CUDA. If you do not have CUDA support, you can modify the script to run on the CPU by changing the device setting.

-----

### 🎯 Project Overview

The core objectives of this project include:

  * Simulating complex input signals containing multiple speakers and background noise.
  * Emulating a microphone array to compute the **Direction of Arrival (DOA)** for sound sources.
  * Applying advanced noise reduction techniques.
  * Separating signals using **Independent Component Analysis (ICA)** and **Beamforming**.
  * Extracting meaningful features from the audio for classification.
  * Utilizing a **self-attention-based BI-LSTM model** for speaker classification.
  * Providing real-time visualizations and detailed analysis of the processed signals.

-----

### 🎵 Input Signal Simulation

Mixed multi-speaker audio signals (male and female) were generated and combined with synthetic noise (e.g., Gaussian noise) using MATLAB's Audio Toolbox to simulate a complex acoustic environment.

-----

### 🎙️ Microphone Array Emulation & DOA

A microphone array was simulated using MATLAB's Phased Array System Toolbox to accurately compute the Direction of Arrival (DOA) for each sound source. This demonstrates the system's ability to localize speakers in a space.

**- Setup:**

  * Male speaker at -20°
  * Female speaker at 0°
  * SNR of 20 dB

**- Computed DOA Results:**

  * Male Speaker: 0°
  * Female Speaker: -20.859°
  * *Note: The results are remarkably accurate, with the system successfully identifying the distinct arrival angles of each speaker.*

-----

### 🔊 Signal Separation & Noise Reduction

The project implements several techniques to isolate and enhance the audio quality.

  * **Signal Separation:** Implemented **Independent Component Analysis (ICA)**, **Blind Source Separation (BSS)**, and **Beamforming** to successfully separate the overlapping audio signals.
  * **Noise Reduction:** Applied **Wiener filtering** and **spectral subtraction** to reduce background noise and improve clarity.

The repository contains the denoised, beamformed, and separated `.wav` files for review.

-----

### 📊 Feature Extraction & Speaker Classification

The final stage of the pipeline involves classifying the individual speakers based on their unique audio characteristics.

  * **Feature Extraction:** **MFCCs (Mel-Frequency Cepstral Coefficients)** and pitch features were extracted to represent speaker traits.
  * **Classification Model:** A **self-attention-based BI-LSTM (Bidirectional Long Short-Term Memory)** model with 1D convolution on the flattened MFCC values was used for classification.

**- Performance Metrics:**
' ![WhatsApp Image 2025-03-25 at 21 32 20_ef7d33fa](https://github.com/user-attachments/assets/8ca077a8-6cf1-45f0-940e-8cd6ee09e1cc)
  * **Classification Accuracy:** ✅ **93.5%**
  * **F1-Score:** ✅ **0.89**

**- Model Performance:**

  * **Confusion Matrix - VoxCeleb1 Dataset:** 
' ![voxceleb1_confusion_matrix](https://github.com/user-attachments/assets/16f6b8f5-1b6f-4e4f-b85b-c11dd5e5815c)
  * **Training History:** 
' ![voxceleb1_training_history](https://github.com/user-attachments/assets/93cc4297-dde0-4244-af56-2d7cce3067d7)

-----

### 📡 Real-Time Analysis & Visualizations

The project includes real-time visualizations of the processed signals, providing a clear and dynamic view of the signal processing and separation in action.
' ![WhatsApp Image 2025-03-25 at 23 17 47_a0155a53](https://github.com/user-attachments/assets/672efe71-f176-4362-8fd0-7b444af1976a)
' ![WhatsApp Image 2025-03-25 at 23 18 28_eb38d162](https://github.com/user-attachments/assets/5bbe28b5-b302-4dc1-83fc-a1bafe419f23)
