function denoised_audio = spec_rem(input_path,output_path,noise_start,noise_end)
% Load the noisy audio file
[input_audio, Fs] = audioread(input_path);

% Define noise estimation range (assume first 1 sec contains only noise)
%noise_start = 0;  % in seconds
%noise_end = 0.1;    % in seconds

% Convert time range to samples
noise_start_sample = floor(noise_start * Fs) + 1;
noise_end_sample = floor(noise_end * Fs);

% Extract noise profile and average it over multiple frames
noise_profile = input_audio(noise_start_sample:noise_end_sample);
noise_profile = reshape(noise_profile, [], 1); % Ensure column vector

% STFT Parameters
n_fft = 1024;      
hop_length = 256;   % 75% overlap for better smoothing
win_length = n_fft;
window = hann(win_length, 'periodic');

% Compute STFT of noisy signal
[S, F, T] = stft(input_audio, Fs, 'Window', window, 'OverlapLength', hop_length, 'FFTLength', n_fft);
[Noise_S, ~, ~] = stft(noise_profile, Fs, 'Window', window, 'OverlapLength', hop_length, 'FFTLength', n_fft);

% Compute noise magnitude spectrum (more frames for better estimate)
Noise_mag = mean(abs(Noise_S), 2);  
Noise_mag = movmean(Noise_mag, 10); % Smoother noise profile estimation

% Compute magnitude and phase of STFT
Mag_S = abs(S);
Phase_S = angle(S);

% **Spectral Floor Limiting & Adaptive Gain**
alpha = 2.5;  % Controls aggressiveness of noise removal
beta = 0.1;   % Prevents sudden unwanted audio bursts

Gain = max((Mag_S - alpha * Noise_mag) ./ Mag_S, beta); % Enforce minimum gain

% **Temporal Smoothing to Remove Sudden Audio Spikes**
Gain_smoothed = 0.9 * Gain + 0.1 * movmean(Gain, 5, 2);

% Apply the gain function
S_denoised = Gain_smoothed .* Mag_S .* exp(1j * Phase_S);

% Perform Inverse STFT
denoised_audio = istft(S_denoised, Fs, 'Window', window, 'OverlapLength', hop_length, 'FFTLength', n_fft);

% Convert to real values and normalize
denoised_audio = real(denoised_audio);
denoised_audio = denoised_audio / max(abs(denoised_audio));

% **Post-Processing: MMSE Wiener Filtering to Remove Remaining Artifacts**
denoised_audio = wiener2(denoised_audio, [7 1]); 

%Amplify the output
denoised_audio = 1000*denoised_audio;

% Save the denoised audio
audiowrite(output_path, denoised_audio, Fs);

% Play the cleaned audio