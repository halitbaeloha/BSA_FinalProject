%% Sleep State Tagging and Hypnogram Script with Integrated Chunk-Based Pre-Processing
% This script performs the following steps:
% 1. Loads an EEG binary file from a specified basepath and basename.
% 2. Reads the EEG data in chunks (to manage memory), applies a 4th-order
%    Butterworth low-pass filter (cutoff = 100 Hz) using zero-phase filtering,
%    and downsamples the data from 32000 Hz to 1000 Hz.
% 3. Smooths the entire EEG signal using a Gaussian window.
% 4. Calculates the RMS of the raw (i.e., unsmoothed) EEG signal per epoch.
% 5. Divides the denoised EEG into 15-minute epochs, computes spectral features,
%    and tags each epoch into one of four sleep states:
%       1 - Wakefulness, 2 - NREM, 3 - REM, 4 - Transition (NREM to REM).
% 6. Visualizes trends via line and scatter plots of spectral ratios.
% 7. Displays a spectrogram of the entire recording along with a hypnogram
%    (sleep state tagging) aligned in time.
% 8. Performs bout analysis of sleep states.

clear; clc;

%% Pre-Processing Section
% Define file location and name.
basepath = 'D:\halit\bsa';    % Modify to your base directory
basename = 'sleepEPhys';      % Base name (without extension)
cd(basepath);
filename = fullfile(basepath, [basename, '.bin']);

% Pre-processing parameters.
Fs_original = 32000;            % Original sampling rate in Hz
Fs_downsampled = 1000;          % Target sampling rate after downsampling in Hz
downsample_factor = Fs_original / Fs_downsampled;  % 32
lp_cutoff_freq = 100;           % Low-pass filter cutoff frequency in Hz
chunk_size = 100000;            % Number of samples per chunk

% Validate cutoff frequency.
if lp_cutoff_freq > Fs_original/2
    warning('Low pass cutoff beyond Nyquist frequency');
end

% Design a 4th-order Butterworth low-pass filter.
[lp_b, lp_a] = butter(4, lp_cutoff_freq/(Fs_original/2), 'low');

% Open the EEG file in binary mode.
fileID = fopen(filename, 'rb');
if fileID == -1
    error('File cannot be opened: %s', filename);
end

% Read and process EEG data chunk-by-chunk.
chunk_counter = 0;
EEG = [];  % Initialize an empty array for the processed EEG.

tic
while ~feof(fileID)
    % Read a chunk as 16-bit integers.
    chunk = fread(fileID, chunk_size, 'int16');
    if isempty(chunk)
        break;
    end
    
    chunk_counter = chunk_counter + 1; 
    fprintf('Processing chunk %d with %d samples\n', chunk_counter, length(chunk));
   
    % Convert to double precision.
    chunk_double = double(chunk);
    
    % Apply zero-phase low-pass filtering.
    chunk_filtered = filtfilt(lp_b, lp_a, chunk_double);
    
    % Downsample the filtered data.
    chunk_downsampled = downsample(chunk_filtered, downsample_factor);
    
    % Append to the main EEG array.
    EEG = [EEG; chunk_downsampled];
    toc
end
fprintf('EEG file created. That took %.2f minutes\n', toc/60)

% Smooth the entire EEG signal with a Gaussian window (window size = 50 samples).
EEG_denoised = smoothdata(EEG, 'gaussian', 50);
% Alternatively, you could use a moving average:
% EEG_denoised = smoothdata(EEG, 'movmean', 50);

%% Calculate EEG RMS on Raw Data (EEG) Per Epoch
% Calculate the root mean square (RMS) for each epoch of the raw EEG.
epoch_duration = 900;  % 15 minutes per epoch (in seconds)
samples_per_epoch = Fs_downsampled * epoch_duration;  % Samples per epoch
num_epochs = floor(length(EEG_denoised) / samples_per_epoch);  % Number of complete epochs
EEG_RMS = zeros(num_epochs, 1);  % Preallocate RMS vector

for i = 1:num_epochs
    % Extract raw epoch data from the downsampled EEG.
    epoch_data = EEG((i-1)*samples_per_epoch + 1 : i*samples_per_epoch);
    % Compute RMS.
    EEG_RMS(i) = sqrt(mean(epoch_data.^2));
end

%% Sleep State Tagging Using Spectral Ratios
% For each epoch, compute the PSD using Welch's method, normalize power
% in defined frequency bands, calculate spectral ratios, and classify sleep states.
% Frequency bands (Hz):
%   Delta: 0.5–4, Theta: 4–8, Alpha: 8–14, Beta: 14–30.
% Sleep state mapping:
%   1 = Wakefulness, 2 = NREM, 3 = REM, 4 = Transition.

sleep_states = zeros(num_epochs, 1);
delta_beta_ratios = zeros(num_epochs, 1);
theta_alpha_ratios = zeros(num_epochs, 1);
delta_theta_ratios = zeros(num_epochs, 1);  % Delta/Theta ratio

% Define frequency bands.
delta_band = [0.5 4];
theta_band = [4 8];
alpha_band = [8 14];
beta_band  = [14 30];

% Define power ratio thresholds.
delta_beta_thresh = 580;    % Threshold for delta/beta ratio
delta_theta_thresh = 45;    % Threshold for delta/theta ratio 
theta_alpha_thresh = 3.5;   % Threshold for theta/alpha ratio

for i = 1:num_epochs
    % Extract epoch data from the denoised EEG.
    epoch_data = EEG_denoised((i-1)*samples_per_epoch + 1 : i*samples_per_epoch);
    
    % Compute PSD using Welch's method with a Hamming window.
    [Pxx, F] = pwelch(epoch_data, hamming(samples_per_epoch), round(0.5*samples_per_epoch), samples_per_epoch, Fs_downsampled);
    
    % --- Modification: Normalize using the power in the sleep-relevant band [0.5 30 Hz] ---
    % Compute total power for normalization over the frequency band of interest.
    norm_band = [0.5 30];  % Define normalization band based on the frequency bands of interest.
    total_power = bandpower(Pxx, F, norm_band, 'psd');
    
    % Compute normalized power in each defined band.
    delta_power = bandpower(Pxx, F, delta_band, 'psd') / total_power;
    theta_power = bandpower(Pxx, F, theta_band, 'psd') / total_power;
    alpha_power = bandpower(Pxx, F, alpha_band, 'psd') / total_power;
    beta_power  = bandpower(Pxx, F, beta_band, 'psd') / total_power;
    
    % Calculate spectral ratios.
    ratio_delta_beta = delta_power / beta_power;
    ratio_theta_alpha = theta_power / alpha_power;
    ratio_delta_theta = delta_power / theta_power;
    ratio_theta_delta = theta_power / delta_power;  % (For reference)
    
    % Store the ratios.
    delta_beta_ratios(i) = ratio_delta_beta;
    theta_alpha_ratios(i) = ratio_theta_alpha;
    delta_theta_ratios(i) = ratio_delta_theta;
    
    % Classify sleep state using defined thresholds.
    if ratio_delta_beta >= delta_beta_thresh && ratio_delta_theta >= delta_theta_thresh  
        sleep_states(i) = 2;  % NREM sleep
    elseif ratio_delta_beta <= delta_beta_thresh && ratio_theta_alpha < theta_alpha_thresh
        sleep_states(i) = 1;  % Wakefulness
    elseif ratio_delta_theta <= delta_theta_thresh && ratio_theta_alpha >= theta_alpha_thresh 
        if i > 1 && sleep_states(i-1) == 3
            sleep_states(i) = 3;  % REM sleep (if previous epoch was REM)
        else
            sleep_states(i) = 4;  % Transition (NREM to REM)
        end
    end
end

% Create a time vector for the midpoints of each epoch (in hours).
time_epochs = ((epoch_duration/2) : epoch_duration : num_epochs*epoch_duration) / 3600;

%% Visualization: PSD, Hypnogram, and Delta/Theta Ratio
figure;
% Top Plot: PSD Over Time
subplot(3,1,1);
window_length = 100000;               % Window length in samples (Note: adjust if needed)
noverlap = round(0.9 * window_length);  % 90% overlap for smoother time resolution
nfft_val = 2048;                      % FFT length
[~, F, T, P] = spectrogram(EEG_denoised, hamming(window_length), noverlap, nfft_val, Fs_downsampled, 'yaxis');

% Restrict frequencies to 0–15 Hz.
freqIdx = F <= 15;
% Convert time vector T from seconds to hours.
T_hours = T / 3600;

% --- Modification: Plot the spectrogram on a logarithmic (dB) scale ---
% Convert power to decibels.
imagesc(T_hours, F(freqIdx), 10*log10(P(freqIdx, :)));
axis xy;
colormap jet;
colorbar;
xlabel('Time (hours)');
ylabel('Frequency (Hz)');
title('Power Spectral Density (PSD) Over Time (log Scale)');

% Middle Plot: Hypnogram
subplot(3,1,2);
plot(time_epochs, sleep_states, 'k', 'LineWidth', 2);
ylim([0.5 4.5]);
yticks([1 2 3 4]);
yticklabels({'Wake', 'NREM', 'REM', 'Transition'});
xlabel('Time (hours)');
ylabel('Sleep State');
title('Hypnogram (Sleep States Over Time)');
grid on;

% Bottom Plot: Delta/Theta Ratio Over Time
subplot(3,1,3);
plot(time_epochs, delta_theta_ratios, 'b', 'LineWidth', 1.5);
xlabel('Time (hours)');
ylabel('Delta/Theta Ratio');
title('Delta/Theta Ratio Over Time');
grid on;

%% Visualization: Continuous Graphs of Spectral Ratios vs. EEG RMS with Ratio Thresholds
% Plot scatter plots of spectral ratios (delta/beta and delta/theta) against EEG RMS.
% Points exceeding the defined thresholds are highlighted.
delta_beta_threshold = 580;   % Threshold for delta/beta ratio
delta_theta_threshold = 55;   % Threshold for delta/theta ratio

figure;
% Top Subplot: Delta/Beta Ratio vs. EEG RMS
subplot(2,1,1);
plot(EEG_RMS, delta_beta_ratios, 'o', 'LineWidth', 2, 'MarkerSize', 4, 'Color', '#80B3FF');
hold on;
aboveIdx = find(delta_beta_ratios > delta_beta_threshold);
plot(EEG_RMS(aboveIdx), delta_beta_ratios(aboveIdx), 'o', 'LineWidth', 2, 'MarkerSize', 4, 'Color', 'r');
xlabel('EEG RMS');
ylabel('Delta/Beta Ratio');
title('Delta/Beta Ratio vs. EEG RMS');
grid on;
yline(delta_beta_threshold, '--k', 'Delta/Beta Threshold');

% Bottom Subplot: Delta/Theta Ratio vs. EEG RMS
subplot(2,1,2);
plot(EEG_RMS, delta_theta_ratios, 'd', 'LineWidth', 2, 'MarkerSize', 4, 'Color', [0 0.4470 0.7410]);
hold on;
aboveIdx = find(delta_theta_ratios > delta_theta_threshold);
plot(EEG_RMS(aboveIdx), delta_theta_ratios(aboveIdx), 'd', 'LineWidth', 2, 'MarkerSize', 4, 'Color', 'r');
xlabel('EEG RMS');
ylabel('Delta/Theta Ratio');
title('Delta/Theta Ratio vs. EEG RMS');
grid on;
yline(delta_theta_threshold, '--k', 'Delta/Theta Threshold');

%% Trends and Scatter Plots of Spectral Ratios only (no RMS)
figure; 
subplot(2,1,1);
plot(time_epochs, delta_beta_ratios, 'LineWidth', 1, 'Color', '#775780');
hold on;
aboveIdx = find(delta_beta_ratios > 550);
d = diff(aboveIdx);
segmentBoundaries = find(d > 1);
startIdx = [aboveIdx(1); aboveIdx(segmentBoundaries+1)];
endIdx = [aboveIdx(segmentBoundaries); aboveIdx(end)];
for k = 1:length(startIdx)
    idxSegment = startIdx(k):endIdx(k);
    plot(time_epochs(idxSegment), delta_beta_ratios(idxSegment), 'LineWidth', 1, 'Color', '#80B3FF');
end
xlabel('Time (hours)');
ylabel('Delta/Beta Ratio');
title('Delta/Beta Ratio over Time');
grid on;

subplot(2,1,2);
plot(time_epochs, theta_alpha_ratios, 'LineWidth', 1);
xlabel('Time (hours)');
ylabel('Theta/Alpha Ratio');
title('Theta/Alpha Ratio over Time');
grid on;

% Scatter plots for delta/theta and delta/beta ratios per epoch.
figure;
subplot(2,1,1);
scatter(time_epochs, delta_theta_ratios, 10, 'filled');
xlabel('Time (hours)');
ylabel('Delta/Theta Ratio');
title('Scatter Plot: Delta/Theta Ratio per Epoch');
grid on;

subplot(2,1,2);
scatter(time_epochs, delta_beta_ratios, 10, 'filled');
xlabel('Time (hours)');
ylabel('Delta/Beta Ratio');
title('Scatter Plot: Delta/Beta Ratio per Epoch');
grid on;

%% Visualization: PSD, Hypnogram, and Delta/Theta Ratio
% For a sensitive PSD analysis, we use a shorter window length (10 seconds) with high overlap.
figure;
% Top Plot: PSD Over Time
subplot(3,1,1);
window_length = 100000;               % 10 seconds (10000 samples at 1000 Hz)
noverlap = round(0.9 * window_length); % 90% overlap for smoother time resolution
nfft_val = 2048;                     % FFT length
[~, F, T, P] = spectrogram(EEG_denoised, hamming(window_length), noverlap, nfft_val, Fs_downsampled, 'yaxis');

% Restrict frequencies to 0–15 Hz.
freqIdx = F <= 15;
% Convert time vector T from seconds to hours.
T_hours = T / 3600;

% Plot the spectrogram using absolute power.
imagesc(T_hours, F(freqIdx), P(freqIdx, :));
axis xy;
colormap jet;
colorbar;
xlabel('Time (hours)');
ylabel('Frequency (Hz)');
title('Power Spectral Density (PSD) Over Time');

% Middle Plot: Hypnogram
subplot(3,1,2);
plot(time_epochs, sleep_states, 'k', 'LineWidth', 2);
ylim([0.5 4.5]);
yticks([1 2 3 4]);
yticklabels({'Wake', 'NREM', 'REM', 'Transition'});
xlabel('Time (hours)');
ylabel('Sleep State');
title('Hypnogram (Sleep States Over Time)');
grid on;

% Bottom Plot: Delta/Theta Ratio Over Time
subplot(3,1,3);
plot(time_epochs, delta_theta_ratios, 'b', 'LineWidth', 1.5);
xlabel('Time (hours)');
ylabel('Delta/Theta Ratio');
title('Delta/Theta Ratio Over Time');
grid on;

%% Sleep State Analysis: Histogram and Duration Calculation with Color-Coded Bars
% Define state labels for display.
state_labels = {'Wake', 'NREM', 'REM', 'Transition'};

% Count the number of epochs per sleep state.
epoch_counts = zeros(4,1);
for s = 1:4
    epoch_counts(s) = sum(sleep_states == s);
end

% Calculate total duration spent in each state.
% Epoch duration is in seconds.
duration_seconds = epoch_counts * epoch_duration; 
duration_minutes = duration_seconds / 60;
duration_hours   = duration_minutes / 60;

% Define distinct colors for each state: [R G B].
colors = [0 0 1;    % Blue for Wake
          0 1 0;    % Green for NREM
          1 0 0;    % Red for REM
          1 0 1];   % Magenta for Transition

% Create a bar chart (histogram) of sleep state epochs with color coding.
figure;
b = bar(1:4, epoch_counts, 'FaceColor', 'flat');
b.CData = colors;
set(gca, 'XTick', 1:4, 'XTickLabel', state_labels);
xlabel('Sleep State');
ylabel('Number of Epochs');
title('Histogram of Sleep State Epochs');
grid on;

% Create a bar chart of total duration (in minutes) per sleep state.
figure;
b2 = bar(1:4, duration_minutes, 'FaceColor', 'flat');
b2.CData = colors;
set(gca, 'XTick', 1:4, 'XTickLabel', state_labels);
xlabel('Sleep State');
ylabel('Duration (minutes)');
title('Total Duration in Minutes per Sleep State');
grid on;

% Display the duration analysis in the command window.
fprintf('\nSleep State Duration Analysis:\n');
for s = 1:4
    fprintf('%s: %d epochs, Total = %.0f sec (%.1f min, %.2f hrs)\n', ...
        state_labels{s}, epoch_counts(s), duration_seconds(s), duration_minutes(s), duration_hours(s));
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%save sleep struct to save your analysis data%
% Save parameters and results for future use.
Sig_psdPerState = struct();
Sig_psdPerState.info.basepath = basepath;    
Sig_psdPerState.info.basename = basename;     
Sig_psdPerState.info.Fs_original = Fs_original;
Sig_psdPerState.info.Fs_downsampled = Fs_downsampled;
Sig_psdPerState.info.downsample_factor = downsample_factor; 
Sig_psdPerState.info.lp_cutoff_freq = lp_cutoff_freq;           
Sig_psdPerState.info.chunk_size = chunk_size;    
Sig_psdPerState.EEG = EEG_denoised;
Sig_psdPerState.EEGrms = EEG_RMS;
Sig_psdPerState.time_epochs = time_epochs;
Sig_psdPerState.epoch_duration = epoch_duration;
Sig_psdPerState.states = sleep_states;
Sig_psdPerState.ratios.delta_beta = delta_beta_ratios;
Sig_psdPerState.ratios.delta_theta = delta_theta_ratios;
Sig_psdPerState.ratios.theta_alpha = theta_alpha_ratios;
Sig_psdPerState.sleepAnalysis.epoch_counts = epoch_counts;
Sig_psdPerState.sleepAnalysis.duration_seconds = duration_seconds;
Sig_psdPerState.sleepAnalysis.duration_minutes = duration_minutes;
Sig_psdPerState.sleepAnalysis.duration_hours = duration_hours;
save('sleep_analysis', 'Sig_psdPerState');