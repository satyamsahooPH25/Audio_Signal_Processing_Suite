' ***************************************
' * Pegasus Cocktail Party *
' ***************************************

' 🚀 To Run
' 1. Replace the files sample_male.wav, sample_female.wav in the folder.
' 2. Install dependencies using:
'    pip install -r requirements.txt
' 3. Run main.wav.
' 4. Ensure you have CUDA support for predict.py. If not, change it to run on 'cpu' and save the changes.

' ***************************************
' * 🎯 Project Overview *
' ***************************************
' This project focuses on multi-speaker audio signal processing using advanced signal processing techniques 
' and deep learning models. The key objectives include:
' - Simulating input signals with mixed multi-speaker audio and noise.
' - Emulating microphone arrays and computing Direction of Arrival (DOA).
' - Applying noise reduction techniques like Wiener filtering.
' - Separating signals using ICA/BSS and beamforming.
' - Extracting features & classification using MFCCs, pitch detection, and BI-LSTM models.
' - Real-time visualization & analysis of the processed signals.

' ***************************************
' * 🎵 Input Signal Simulation *
' ***************************************
' Generated and mixed multi-speaker (Male, Female) audio signals with synthetic or real-world noise (Gaussian Noise) 
' using MATLAB’s Audio Toolbox.

' [Image]
' ![WhatsApp Image 2025-03-25 at 22 36 52_bab79a8b](https://github.com/user-attachments/assets/6a5454fa-53c1-4661-9f5d-858262d42452)

' ***************************************
' * 🎙️ Microphone Array Emulation *
' ***************************************
' Simulated microphone arrays using the Phased Array System Toolbox to compute DOA (Direction of Arrival) for sound sources.

' - Setup:
'   - Male speaker at -20°, Female speaker at 0°
'   - SNR of 20 dB
' - Computed DOA results:
'   - Male Speaker: 0°
'   - Female Speaker: -20.859° (showcasing remarkable accuracy)

' [Images]
' ![WhatsApp Image 2025-03-25 at 22 47 21_9a0c1ad4](https://github.com/user-attachments/assets/812fd0bd-2cae-47e8-8390-8060542c05f9)
' ![WhatsApp Image 2025-03-25 at 23 31 30_2b0eb1a4](https://github.com/user-attachments/assets/27c74867-c917-4f25-b4a9-d6cb62ab2b9e)

' ***************************************
' * 🔊 Signal Separation *
' ***************************************
' Implemented Independent Component Analysis (ICA), Blind Source Separation (BSS), and Beamforming to 
' separate overlapping audio signals and measure the quality of separation.

' 📌 Denoised and beamformed .wav files, along with computed DOA results, are uploaded in the repository.

' ***************************************
' * 🎛️ Noise Reduction *
' ***************************************
' Applied Wiener filtering and spectral subtraction to enhance audio quality by reducing background noise.

' 📌 Denoised male and female audio files are available in the repository.

' ***************************************
' * 📊 Feature Extraction & Classification *
' ***************************************
' - Extracted MFCCs and pitch features to represent speaker characteristics.
' - Used a self-attention-based BI-LSTM model with 1D convolution on flattened MFCC values.
' - Achieved:
'   - ✅ Classification Accuracy: 93.5%
'   - ✅ F1-Score: 0.89

' [Performance Visualizations]
' ![WhatsApp Image 2025-03-25 at 21 32 20_ef7d33fa](https://github.com/user-attachments/assets/8ca077a8-6cf1-45f0-940e-8cd6ee09e1cc)

' [Confusion Matrix - VoxCeleb1 Dataset]
' ![voxceleb1_confusion_matrix](https://github.com/user-attachments/assets/16f6b8f5-1b6f-4e4f-b85b-c11dd5e5815c)

' [Training History]
' ![voxceleb1_training_history](https://github.com/user-attachments/assets/93cc4297-dde0-4244-af56-2d7cce3067d7)

' ***************************************
' * 📡 Visualization & Real-Time Analysis *
' ***************************************
' ![WhatsApp Image 2025-03-25 at 23 17 47_a0155a53](https://github.com/user-attachments/assets/672efe71-f176-4362-8fd0-7b444af1976a)
' ![WhatsApp Image 2025-03-25 at 23 18 28_eb38d162](https://github.com/user-attachments/assets/5bbe28b5-b302-4dc1-83fc-a1bafe419f23)

' ***************************************
' * 🤝 Contributors *
' ***************************************
' - Satyam Swayamjeet Sahoo

' 📌 Feel free to contribute and improve this project!
