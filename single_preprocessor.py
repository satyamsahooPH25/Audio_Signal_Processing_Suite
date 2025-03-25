import os
import librosa
import numpy as np
import argparse

def pad_or_truncate(array, target_length=130):
    """Standardize array length"""
    current_length = array.shape[1]
    if current_length > target_length:
        return array[:, :target_length]
    elif current_length < target_length:
        pad_width = ((0, 0), (0, target_length - current_length))
        return np.pad(array, pad_width, mode='constant')
    return array

def preprocess_audio(file_path, sr=22050, duration=3, target_length=130):
    """Extract audio features with fixed length output"""
    try:
        # Load and pre-emphasize audio
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        emphasized_signal = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # Window parameters
        frame_length = int(sr * 0.025)  # 25ms window
        hop_length = int(sr * 0.01)    # 10ms hop
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=emphasized_signal,
            sr=sr,
            n_mfcc=23,
            dct_type=2,
            norm='ortho',
            lifter=22,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=40,
            center=False,
            window=np.hanning(frame_length)
        )
        # Ensure all features have same length
        min_length = 299
        mfcc = mfcc[:, :min_length]
        features = np.vstack([mfcc])
        # Pad or truncate to fixed length
        features = pad_or_truncate(features, target_length)
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def pathmain(input_file, output_file):
    features = preprocess_audio(input_file)
    if features is not None:
        np.save(output_file, features)
        print(f"Processed file saved as {output_file}")
    else:
        print("Failed to process the file.")


pathmain("denoised_source_1.wav", "denoised_source_1.npy")
pathmain("denoised_source_2.wav", "denoised_source_2.npy")
