
warning('off', 'all');
%% Define File Paths
input_wav1 = ".\sample_male.wav";
output_wav1 = ".\sample_male_gauss.wav";
input_wav2 = ".\samplefemale.mp3";
output_wav2 = ".\sample_female_gauss.wav";

SNR_db = 20; % Adjust signal-to-noise ratio if needed

% Read the input audio files
[audio1, sample_rate1] = audioread(input_wav1);
[audio2, sample_rate2] = audioread(input_wav2);

% Normalize audio if needed
audio1 = audio1 / max(abs(audio1));
audio2 = audio2 / max(abs(audio2));

% Calculate signal power
signal_power1 = mean(audio1.^2);
signal_power2 = mean(audio2.^2);

% Compute noise power and standard deviation
noise_power1 = signal_power1 / (10^(SNR_db / 10));
noise_power2 = signal_power2 / (10^(SNR_db / 10));
noise_std1 = sqrt(noise_power1);
noise_std2 = sqrt(noise_power2);

% Generate Gaussian noise
noise1 = noise_std1 * randn(size(audio1));
noise2 = noise_std2 * randn(size(audio2));

% Add noise and prevent clipping
noisy_audio1 = max(min(audio1 + noise1, 1), -1);
noisy_audio2 = max(min(audio2 + noise2, 1), -1);

% Save the noisy audio
audiowrite(output_wav1, noisy_audio1, sample_rate1);
audiowrite(output_wav2, noisy_audio2, sample_rate2);

fprintf('Generated noisy files: %s, %s\n', output_wav1, output_wav2);



%% Load Noisy Source Signals
sourceFiles = {output_wav1, output_wav2};

sources = cell(1, length(sourceFiles));
desired_fs = 44100;

for i = 1:length(sourceFiles)
    if exist(sourceFiles{i}, 'file')
        [audio, fs] = audioread(sourceFiles{i});
        disp(fs);
        % Resample if needed
        if fs ~= desired_fs
            audio = resample(audio, desired_fs, fs);
            fs = desired_fs;
        end
        
        sources{i} = audio(:, 1); % Use single channel
        sources{i} = sources{i} / max(abs(sources{i})); % Normalize
    else
        error("File not found: %s", sourceFiles{i});
    end
end

disp("Noisy audio files loaded successfully.");

%% Define True DOA Angles
true_angles = [0 -20; 0 0];
disp('True DOA Angles (Azimuthal):');
disp(true_angles(1,:));

%% Simulation of the Microphone Array
num_mics = 9;
mic_Spacing = 0.5;
c = 340;

microphone = phased.OmnidirectionalMicrophoneElement('FrequencyRange', [0 20e3]);
micArray = phased.ULA(num_mics, mic_Spacing, 'Element', microphone);

%% Ensure Same Duration for Signals
min_samples = min(cellfun(@length, sources));
for i = 1:length(sources)
    sources{i} = sources{i}(1:min_samples);
end

signal = cell2mat(sources);
num_sources = size(signal, 2);

%% Simulate Received Signals at Microphone Array
collector = phased.WidebandCollector('Sensor', micArray, 'PropagationSpeed', c, ...
    'SampleRate', fs, 'NumSubbands', 2000, 'ModulatedInput', false,'Wavefront','Plane','Polarization','None');
sig_1 = collector(signal(:,1), true_angles(:,1));
sig_2 = collector(signal(:,2), true_angles(:,2));

sig = sig_1 + sig_2;

%% Add Gaussian White Noise (SNR = 20 dB)
SNR_dB = 60;
signal_power = mean(abs(sig).^2, 'all'); 
noise_power = signal_power / (10^(SNR_dB/10));
noise = sqrt(noise_power) * randn(size(sig));
noisy_signal = sig;

%% Mix Signals After Normalization
mixed_signal1 = sum(noisy_signal, 2);  
mixed_signal = mixed_signal1 / max(abs(mixed_signal1));

%% Save Mixed Signal
output_folder = ".\";
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
audiowrite(fullfile(output_folder, 'mixed_signal.mp3'), real(mixed_signal), fs);

%% DOA Estimation using MUSIC Algorithm
fc = 300e6;
lambda = physconst('LightSpeed')/fc;
pos = getElementPosition(micArray)/lambda;
Nsamp = 1000;

new_signal = sensorsig(pos,Nsamp,true_angles);

musicspatialspect = phased.MUSICEstimator('SensorArray', micArray, ...
    'OperatingFrequency', 300e6, 'ScanAngles', -90:90, ...
    'DOAOutputPort', true, 'NumSignalsSource', 'Property', 'NumSignals', num_sources);

[~, estimatedDOA] = musicspatialspect(new_signal);

%% Print Estimated DOA Values
disp('Estimated DOA Angles:');
estimatedDOA = broadside2az(estimatedDOA, [2,2]);
disp(estimatedDOA);

%% Wideband Frost Beamforming
beamformedSignals = zeros(size(sig, 1), length(estimatedDOA));
beamformedFiles = cell(1, length(estimatedDOA));

for i = 1:length(estimatedDOA)
    beamformer = phased.FrostBeamformer('SensorArray', micArray, ...
        'PropagationSpeed', c, 'SampleRate', fs, 'DirectionSource', 'Input port','DiagonalLoadingFactor', 0.01);
    beamformedSignals(:, i) = real(beamformer(sig,[estimatedDOA(i);0]));
    beamformedSignals(:, i) = beamformedSignals(:, i) / max(abs(beamformedSignals(:, i))); % Normalize

    % Save beamformed signals
    beamformedFiles{i} = fullfile(output_folder, sprintf('beamformed_source_%d_DOA_%d.wav', i, round(estimatedDOA(i))));
    audiowrite(beamformedFiles{i}, beamformedSignals(:, i), fs);
    disp(['Saved beamformed signal: ', beamformedFiles{i}]);
end

%% Call Python Script for Denoising
denoisedFiles = cell(1, length(beamformedFiles));


for i = 1:length(beamformedFiles)
    denoisedFiles{i} = fullfile(output_folder, sprintf('denoised_source_%d.wav', i));
    spectral_rem(beamformedFiles{i}, denoisedFiles{i},0,0.1); 
end

system(".\single_preprocessor.py");

disp('Beamforming and Denoising completed successfully.');

system(".\predict.py");