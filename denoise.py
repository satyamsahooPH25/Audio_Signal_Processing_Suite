import sys
import numpy as np
import librosa
from scipy.io import wavfile
import noisereduce as nr

def denoise(input_file, output_file, noise_start=0, noise_end=1):
    """
    Denoise an audio file and save the output.
    """
    print(f"Processing: {input_file}")

    # Load audio file
    audio, sr = librosa.load(input_file, sr=None, mono=True)

    # Extract noise profile
    noise = audio[int(noise_start*sr):int(noise_end*sr)]

    # Apply noise reduction
    denoised = nr.reduce_noise(y=audio, y_noise=noise, sr=sr, prop_decrease=0.85, stationary=False)

    # Scale back to 16-bit PCM
    denoised = np.int16(denoised * (32767 / np.max(np.abs(denoised))))

    # Save denoised output
    wavfile.write(output_file, sr, denoised)
    print(f"Denoised file saved: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python denoise.py <input_wav> <output_wav>")
        sys.exit(1)

    input_wav = sys.argv[1]
    output_wav = sys.argv[2]
    denoise(input_wav, output_wav)
