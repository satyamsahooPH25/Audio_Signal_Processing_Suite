import numpy as np
from scipy.io import wavfile
import sys

def add_gaussian_noise(input_file, output_file, SNR_db=20):
    """
    Add Gaussian (normal) noise to an audio file
    :param input_file: Input WAV path
    :param output_file: Output WAV path
    :param SNR_db: Signal to Noise ratio in decibels
    """
    # Read audio
    sample_rate, data = wavfile.read(input_file)
    audio = data.astype(np.float32) / (2**15)

    # Calculating Signal Power
    signal_power = np.mean(audio**2)

    # Compute noise power and standard deviation (scale)
    noise_power = signal_power / (10 ** (SNR_db / 10))
    scale = np.sqrt(noise_power)

    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=scale, size=audio.shape)

    # Add noise to the signal and prevent clipping
    noisy_audio = np.clip(audio + noise, -1.0, 1.0)

    # Save noisy audio
    wavfile.write(output_file, sample_rate, (noisy_audio * (2**15)).astype(np.int16))

    print(f"Generated noisy file: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python add_gaussian_noise.py <input1> <output1> <input2> <output2> [SNR_db]")
        sys.exit(1)

    # Read arguments from MATLAB
    input_wav1 = sys.argv[1]
    output_wav1 = sys.argv[2]
    input_wav2 = sys.argv[3]
    output_wav2 = sys.argv[4]
    SNR_db = float(sys.argv[5]) if len(sys.argv) > 5 else 20

    # Process both audio files
    add_gaussian_noise(input_wav1, output_wav1, SNR_db)
    add_gaussian_noise(input_wav2, output_wav2, SNR_db)
