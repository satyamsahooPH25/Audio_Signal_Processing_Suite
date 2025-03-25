Pegasus_Cocktail-Party_1

TO RUN- Replace files sample_male.wav, sample_female.wav in the folder along with pip installation of requirements.txt and run main.wav, make sure you have CUDA support for predict.py else change to 'cpu' and save it. 

TASKS-
Input Signal Simulation - Generated and mixed multi-speaker (Male,Female) audio signals with synthetic or real-world noise (Gaussian Noise) using MATLAB’s Audio Toolbox.
![WhatsApp Image 2025-03-25 at 22 36 52_bab79a8b](https://github.com/user-attachments/assets/6a5454fa-53c1-4661-9f5d-858262d42452)

Microphone Array Emulation- Simulated microphone arrays using the Phased Array System Toolbox, compute DOA for sound sources. (We have used Male and Female speaker at -20 degrees and 0 degrees respectively, file names - sample_male.wav, sample_female.wav with SNR 20db, and resultant DOA computed are 0 degrees and -20.859 degrees showcasing remarkable accuracy )
![WhatsApp Image 2025-03-25 at 22 47 21_9a0c1ad4](https://github.com/user-attachments/assets/812fd0bd-2cae-47e8-8390-8060542c05f9)
![WhatsApp Image 2025-03-25 at 23 31 30_2b0eb1a4](https://github.com/user-attachments/assets/27c74867-c917-4f25-b4a9-d6cb62ab2b9e)

Signal Separation- Implemented ICA/BSS and Beamforming techniques to separate audio signals and measure separation quality.(The denoised and Beamformed result .wav files along with DOA are uploaded in the repository)

Noise Reduction- Applied Wiener filtering and spectral subtraction to enhance audio quality. (Denoised audio for male and female are saved in the repository)

Feature Extraction & Classification - Extract MFCCs, pitch features, and classify speakers using custom self-attention based BI-LSTM models after 1-D Convolution of flattened MFCC Values
We have achieved a classification accuracy of 93.5% and F1-Score of 0.89
![WhatsApp Image 2025-03-25 at 21 32 20_ef7d33fa](https://github.com/user-attachments/assets/8ca077a8-6cf1-45f0-940e-8cd6ee09e1cc)


![WhatsApp Image 2025-03-25 at 23 17 47_a0155a53](https://github.com/user-attachments/assets/672efe71-f176-4362-8fd0-7b444af1976a)
![WhatsApp Image 2025-03-25 at 23 18 28_eb38d162](https://github.com/user-attachments/assets/5bbe28b5-b302-4dc1-83fc-a1bafe419f23)


