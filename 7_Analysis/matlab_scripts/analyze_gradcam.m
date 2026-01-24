% Analyze Grad-CAM Results
% This script reads Grad-CAM heatmaps from DualEEG model analysis
% and generates visualizations:
% 1. Three-class Heatmap Comparison
% 2. Difference Maps (pairwise class differences)
% 3. Frequency Profile (importance by frequency band)
% 4. Temporal Profile (importance over time)
% 5. Band Statistics Bar Chart (Delta/Theta/Alpha/Beta/Gamma)
% 6. 3D Surface Plot
%
% Grad-CAM is computed on the Spectrogram representation:
%   - 64 frequency bins (rows) from STFT
%   - 64 time steps (columns) from STFT
%
% Data source: gradcam/ from Python analysis pipeline

clear; clc; close all;


%% ========================================================================
% Configuration
% =========================================================================

% Get the directory of the current script
script_dir = fileparts(mfilename('fullpath'));

% Define paths
data_dir = fullfile(script_dir, '..', 'raw_result', 'dualEEG_old_eeg', 'gradcam');
output_dir = fullfile(script_dir, '..', 'figures', 'gradcam_matlab');

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

% Class names and colors
classes = {'Single', 'Competition', 'Cooperation'};
colors_class = [0.55, 0.63, 0.80;   % Single - Blue
                0.99, 0.55, 0.38;   % Competition - Orange
                0.40, 0.76, 0.65];  % Cooperation - Green

% Spectrogram parameters (from model config)
sampling_rate = 256;  % Hz
n_fft = 128;
hop_length = 64;
freq_bins = 64;
time_steps = 64;

% Frequency resolution
freq_resolution = sampling_rate / n_fft;  % Hz per bin
max_freq = freq_bins * freq_resolution;   % Maximum frequency

% Time resolution (approximate)
% Original signal: 1024 samples at 256 Hz = 4 seconds
% STFT with hop_length=64: ~16 time frames per second
time_resolution = hop_length / sampling_rate;  % seconds per step
total_time = time_steps * time_resolution;     % Total time span

% Define frequency bands (in bins)
% Frequency = bin_index * freq_resolution
bands = struct();
bands.names = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'};
bands.ranges_hz = {[0.5, 4], [4, 8], [8, 13], [13, 30], [30, 45]};
bands.colors = [0.6, 0.2, 0.8;    % Delta - Purple
                0.2, 0.6, 0.8;    % Theta - Cyan
                0.2, 0.8, 0.2;    % Alpha - Green
                0.8, 0.8, 0.2;    % Beta - Yellow
                0.8, 0.4, 0.2];   % Gamma - Orange

% Convert Hz ranges to bin indices
bands.bin_ranges = cell(1, 5);
for i = 1:5
    low_bin = max(1, floor(bands.ranges_hz{i}(1) / freq_resolution) + 1);
    high_bin = min(freq_bins, ceil(bands.ranges_hz{i}(2) / freq_resolution));
    bands.bin_ranges{i} = [low_bin, high_bin];
end

fprintf('\n=== Grad-CAM Analysis ===\n');
fprintf('Spectrogram parameters:\n');
fprintf('  Frequency resolution: %.2f Hz/bin\n', freq_resolution);
fprintf('  Time resolution: %.3f s/step\n', time_resolution);
fprintf('  Total time span: %.2f s\n', total_time);
fprintf('\nFrequency band mappings:\n');
for i = 1:5
    fprintf('  %s: %.1f-%.1f Hz (bins %d-%d)\n', ...
        bands.names{i}, bands.ranges_hz{i}(1), bands.ranges_hz{i}(2), ...
        bands.bin_ranges{i}(1), bands.bin_ranges{i}(2));
end


%% ========================================================================
% Load Data
% =========================================================================

fprintf('\nLoading Grad-CAM data...\n');

gradcam = cell(1, 3);
for i = 1:3
    filepath = fullfile(data_dir, sprintf('gradcam_%s.csv', classes{i}));
    if exist(filepath, 'file')
        gradcam{i} = readmatrix(filepath);
        fprintf('  Loaded %s: %dx%d\n', classes{i}, size(gradcam{i}, 1), size(gradcam{i}, 2));
    else
        error('File not found: %s', filepath);
    end
end

% Create frequency and time axes
freq_axis = (0:freq_bins-1) * freq_resolution;  % Hz
time_axis = (0:time_steps-1) * time_resolution * 1000;  % ms


%% ========================================================================
% 1. Three-class Heatmap Comparison
% =========================================================================

fprintf('\n--- 1. Generating Three-class Heatmap Comparison ---\n');

fig1 = figure('Name', 'Grad-CAM Heatmap Comparison', ...
    'Color', 'w', 'Position', [50, 100, 1400, 450]);

% Find global color limits
all_vals = [gradcam{1}(:); gradcam{2}(:); gradcam{3}(:)];
clim_val = [min(all_vals), max(all_vals)];

for i = 1:3
    subplot(1, 3, i);

    % Plot heatmap (origin='lower' equivalent: flipud or use axis direction)
    imagesc(time_axis, freq_axis, gradcam{i});
    set(gca, 'YDir', 'normal');  % Frequency increases upward
    colormap(gca, jet);
    clim(clim_val);
    colorbar;

    % Add frequency band boundaries
    hold on;
    for b = 1:5
        yline(bands.ranges_hz{b}(2), '--w', 'LineWidth', 1, 'Alpha', 0.7);
    end
    hold off;

    % Labels
    title(classes{i}, 'FontSize', 14, 'FontWeight', 'bold', 'Color', colors_class(i,:));
    xlabel('Time (ms)', 'FontSize', 11);
    ylabel('Frequency (Hz)', 'FontSize', 11);
    ylim([0, 50]);  % Focus on 0-50 Hz (EEG relevant range)
end

sgtitle('Grad-CAM: Time-Frequency Importance by Class', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig1, fullfile(output_dir, 'gradcam_heatmap_comparison.png'));
fprintf('Saved: gradcam_heatmap_comparison.png\n');


%% ========================================================================
% 2. Difference Maps
% =========================================================================

fprintf('\n--- 2. Generating Difference Maps ---\n');

fig2 = figure('Name', 'Grad-CAM Difference Maps', ...
    'Color', 'w', 'Position', [50, 100, 1400, 450]);

% Create diverging colormap
n = 256;
half = n / 2;
blue_to_white = [linspace(0.2, 1, half)', linspace(0.4, 1, half)', linspace(0.8, 1, half)'];
white_to_red = [linspace(1, 0.9, half)', linspace(1, 0.3, half)', linspace(1, 0.3, half)'];
diverging_cmap = [blue_to_white; white_to_red];

% Difference pairs
diff_pairs = {
    {2, 1, 'Competition - Single'}, ...
    {3, 1, 'Cooperation - Single'}, ...
    {3, 2, 'Cooperation - Competition'}
};

for p = 1:3
    idx1 = diff_pairs{p}{1};
    idx2 = diff_pairs{p}{2};
    title_str = diff_pairs{p}{3};

    diff_map = gradcam{idx1} - gradcam{idx2};

    subplot(1, 3, p);
    imagesc(time_axis, freq_axis, diff_map);
    set(gca, 'YDir', 'normal');
    colormap(gca, diverging_cmap);

    % Symmetric color limits
    max_abs = max(abs(diff_map(:)));
    clim([-max_abs, max_abs]);
    colorbar;

    % Add frequency band boundaries
    hold on;
    for b = 1:5
        yline(bands.ranges_hz{b}(2), '--k', 'LineWidth', 1, 'Alpha', 0.5);
    end
    hold off;

    title(title_str, 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Time (ms)', 'FontSize', 11);
    ylabel('Frequency (Hz)', 'FontSize', 11);
    ylim([0, 50]);
end

sgtitle('Grad-CAM Difference Maps (Red=Higher, Blue=Lower)', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig2, fullfile(output_dir, 'gradcam_difference_maps.png'));
fprintf('Saved: gradcam_difference_maps.png\n');


%% ========================================================================
% 3. Frequency Profile
% =========================================================================

fprintf('\n--- 3. Generating Frequency Profile ---\n');

fig3 = figure('Name', 'Grad-CAM Frequency Profile', ...
    'Color', 'w', 'Position', [100, 100, 1000, 500]);

% Average over time axis (columns) to get frequency profile
freq_profiles = zeros(freq_bins, 3);
for i = 1:3
    freq_profiles(:, i) = mean(gradcam{i}, 2);
end

subplot(1, 2, 1);
hold on;
for i = 1:3
    plot(freq_axis, freq_profiles(:, i), '-', 'Color', colors_class(i, :), ...
        'LineWidth', 2, 'DisplayName', classes{i});
end

% Add band shading
y_max = max(freq_profiles(:)) * 1.1;
for b = 1:5
    low_hz = bands.ranges_hz{b}(1);
    high_hz = bands.ranges_hz{b}(2);
    fill([low_hz, high_hz, high_hz, low_hz], [0, 0, y_max, y_max], ...
        bands.colors(b, :), 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
end
hold off;

xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Mean Grad-CAM Importance', 'FontSize', 12);
title('Frequency Profile (averaged over time)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast');
xlim([0, 50]);
grid on;

% Add band labels at top
subplot(1, 2, 2);
hold on;
for i = 1:3
    plot(freq_axis, freq_profiles(:, i), '-', 'Color', colors_class(i, :), ...
        'LineWidth', 2);
end
hold off;

xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Mean Grad-CAM Importance', 'FontSize', 12);
title('Frequency Profile with Band Labels', 'FontSize', 14, 'FontWeight', 'bold');
xlim([0, 50]);
grid on;

% Add band labels
for b = 1:5
    mid_hz = mean(bands.ranges_hz{b});
    text(mid_hz, max(freq_profiles(:)) * 1.05, bands.names{b}, ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
        'Color', bands.colors(b, :));
end

sgtitle('Grad-CAM: Frequency Importance Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig3, fullfile(output_dir, 'gradcam_frequency_profile.png'));
fprintf('Saved: gradcam_frequency_profile.png\n');


%% ========================================================================
% 4. Temporal Profile
% =========================================================================

fprintf('\n--- 4. Generating Temporal Profile ---\n');

fig4 = figure('Name', 'Grad-CAM Temporal Profile', ...
    'Color', 'w', 'Position', [100, 100, 1000, 500]);

% Average over frequency axis (rows) to get temporal profile
temp_profiles = zeros(time_steps, 3);
for i = 1:3
    temp_profiles(:, i) = mean(gradcam{i}, 1);
end

subplot(1, 2, 1);
hold on;
for i = 1:3
    plot(time_axis, temp_profiles(:, i), '-', 'Color', colors_class(i, :), ...
        'LineWidth', 2, 'DisplayName', classes{i});
end
hold off;

xlabel('Time (ms)', 'FontSize', 12);
ylabel('Mean Grad-CAM Importance', 'FontSize', 12);
title('Temporal Profile (averaged over frequency)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast');
grid on;

% Smoothed version
subplot(1, 2, 2);
hold on;
for i = 1:3
    smooth_profile = movmean(temp_profiles(:, i), 5);
    plot(time_axis, smooth_profile, '-', 'Color', colors_class(i, :), ...
        'LineWidth', 2, 'DisplayName', classes{i});
end

% Find peaks
for i = 1:3
    smooth_profile = movmean(temp_profiles(:, i), 5);
    [pks, locs] = findpeaks(smooth_profile, 'MinPeakProminence', max(smooth_profile)*0.1);
    if ~isempty(pks)
        plot(time_axis(locs), pks, 'v', 'Color', colors_class(i, :), ...
            'MarkerSize', 8, 'MarkerFaceColor', colors_class(i, :), ...
            'HandleVisibility', 'off');
    end
end
hold off;

xlabel('Time (ms)', 'FontSize', 12);
ylabel('Mean Grad-CAM Importance', 'FontSize', 12);
title('Temporal Profile (smoothed, peaks marked)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast');
grid on;

sgtitle('Grad-CAM: Temporal Importance Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig4, fullfile(output_dir, 'gradcam_temporal_profile.png'));
fprintf('Saved: gradcam_temporal_profile.png\n');


%% ========================================================================
% 5. Band Statistics Bar Chart
% =========================================================================

fprintf('\n--- 5. Generating Band Statistics Bar Chart ---\n');

fig5 = figure('Name', 'Grad-CAM Band Statistics', ...
    'Color', 'w', 'Position', [100, 100, 1000, 600]);

% Compute mean importance for each band and class
band_stats = zeros(5, 3);  % 5 bands x 3 classes
band_stds = zeros(5, 3);

for b = 1:5
    bin_range = bands.bin_ranges{b}(1):bands.bin_ranges{b}(2);
    for i = 1:3
        band_data = gradcam{i}(bin_range, :);
        band_stats(b, i) = mean(band_data(:));
        band_stds(b, i) = std(band_data(:));
    end
end

% --- 5a: Grouped bar chart ---
subplot(2, 2, 1);
b = bar(band_stats);
for i = 1:3
    b(i).FaceColor = colors_class(i, :);
end
set(gca, 'XTickLabel', bands.names);
xlabel('Frequency Band', 'FontSize', 11);
ylabel('Mean Grad-CAM Importance', 'FontSize', 11);
title('Band Importance by Class', 'FontSize', 13, 'FontWeight', 'bold');
legend(classes, 'Location', 'northeast');
grid on;

% --- 5b: Normalized (per class) ---
subplot(2, 2, 2);
band_stats_norm = band_stats ./ sum(band_stats, 1);  % Normalize per class
b2 = bar(band_stats_norm);
for i = 1:3
    b2(i).FaceColor = colors_class(i, :);
end
set(gca, 'XTickLabel', bands.names);
xlabel('Frequency Band', 'FontSize', 11);
ylabel('Proportion of Total Importance', 'FontSize', 11);
title('Relative Band Importance (normalized)', 'FontSize', 13, 'FontWeight', 'bold');
legend(classes, 'Location', 'northeast');
grid on;

% --- 5c: Band-focused view ---
subplot(2, 2, 3);
band_stats_t = band_stats';  % Transpose: 3 classes x 5 bands
b3 = bar(band_stats_t);
for i = 1:5
    b3(i).FaceColor = bands.colors(i, :);
end
set(gca, 'XTickLabel', classes);
xlabel('Class', 'FontSize', 11);
ylabel('Mean Grad-CAM Importance', 'FontSize', 11);
title('Class Importance by Band', 'FontSize', 13, 'FontWeight', 'bold');
legend(bands.names, 'Location', 'northeast');
grid on;

% --- 5d: Heatmap ---
subplot(2, 2, 4);
imagesc(band_stats);
colormap(gca, parula);
colorbar;
set(gca, 'XTick', 1:3, 'XTickLabel', classes);
set(gca, 'YTick', 1:5, 'YTickLabel', bands.names);
xlabel('Class', 'FontSize', 11);
ylabel('Frequency Band', 'FontSize', 11);
title('Band × Class Importance Heatmap', 'FontSize', 13, 'FontWeight', 'bold');

% Add values as text
for b = 1:5
    for c = 1:3
        text(c, b, sprintf('%.2e', band_stats(b, c)), ...
            'HorizontalAlignment', 'center', 'Color', 'w', ...
            'FontSize', 8, 'FontWeight', 'bold');
    end
end

sgtitle('Grad-CAM: Frequency Band Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig5, fullfile(output_dir, 'gradcam_band_statistics.png'));
fprintf('Saved: gradcam_band_statistics.png\n');

% Export band statistics
band_table = array2table(band_stats, 'VariableNames', classes, 'RowNames', bands.names);
writetable(band_table, fullfile(output_dir, 'gradcam_band_stats.csv'), 'WriteRowNames', true);
fprintf('Saved: gradcam_band_stats.csv\n');


%% ========================================================================
% 6. 3D Surface Plot
% =========================================================================

fprintf('\n--- 6. Generating 3D Surface Plot ---\n');

fig6 = figure('Name', 'Grad-CAM 3D Surface', ...
    'Color', 'w', 'Position', [50, 50, 1500, 500]);

[T, F] = meshgrid(time_axis, freq_axis);

for i = 1:3
    subplot(1, 3, i);

    % Limit to EEG-relevant frequency range (0-50 Hz)
    freq_limit = find(freq_axis <= 50, 1, 'last');

    surf(T(1:freq_limit, :), F(1:freq_limit, :), gradcam{i}(1:freq_limit, :), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.9);

    colormap(gca, jet);
    colorbar;

    xlabel('Time (ms)', 'FontSize', 10);
    ylabel('Frequency (Hz)', 'FontSize', 10);
    zlabel('Importance', 'FontSize', 10);
    title(classes{i}, 'FontSize', 13, 'FontWeight', 'bold', 'Color', colors_class(i, :));

    view(45, 30);
    grid on;
end

sgtitle('Grad-CAM: 3D Time-Frequency Importance Surface', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig6, fullfile(output_dir, 'gradcam_3d_surface.png'));
fprintf('Saved: gradcam_3d_surface.png\n');


%% ========================================================================
% 7. Summary Figure (Publication-ready)
% =========================================================================

fprintf('\n--- 7. Generating Summary Figure ---\n');

fig7 = figure('Name', 'Grad-CAM Summary', ...
    'Color', 'w', 'Position', [50, 50, 1600, 1000]);

% --- Panel A: Three heatmaps ---
for i = 1:3
    subplot(3, 4, i);
    imagesc(time_axis, freq_axis, gradcam{i});
    set(gca, 'YDir', 'normal');
    colormap(gca, jet);
    clim(clim_val);
    if i == 3
        colorbar;
    end
    title(classes{i}, 'FontSize', 11, 'FontWeight', 'bold', 'Color', colors_class(i, :));
    xlabel('Time (ms)');
    ylabel('Freq (Hz)');
    ylim([0, 50]);
    set(gca, 'FontSize', 8);
end

% --- Panel B: Difference (Coop - Comp) ---
subplot(3, 4, 4);
diff_coop_comp = gradcam{3} - gradcam{2};
imagesc(time_axis, freq_axis, diff_coop_comp);
set(gca, 'YDir', 'normal');
colormap(gca, diverging_cmap);
max_abs = max(abs(diff_coop_comp(:)));
clim([-max_abs, max_abs]);
colorbar;
title('Coop - Comp', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Time (ms)');
ylabel('Freq (Hz)');
ylim([0, 50]);
set(gca, 'FontSize', 8);

% --- Panel C: Frequency profile ---
subplot(3, 4, 5:6);
hold on;
for i = 1:3
    plot(freq_axis, freq_profiles(:, i), '-', 'Color', colors_class(i, :), ...
        'LineWidth', 2);
end
hold off;
xlabel('Frequency (Hz)');
ylabel('Importance');
title('Frequency Profile', 'FontSize', 12, 'FontWeight', 'bold');
legend(classes, 'Location', 'northeast', 'FontSize', 8);
xlim([0, 50]);
grid on;
set(gca, 'FontSize', 9);

% --- Panel D: Temporal profile ---
subplot(3, 4, 7:8);
hold on;
for i = 1:3
    plot(time_axis, temp_profiles(:, i), '-', 'Color', colors_class(i, :), ...
        'LineWidth', 2);
end
hold off;
xlabel('Time (ms)');
ylabel('Importance');
title('Temporal Profile', 'FontSize', 12, 'FontWeight', 'bold');
legend(classes, 'Location', 'northeast', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 9);

% --- Panel E: Band bar chart ---
subplot(3, 4, 9:10);
b = bar(band_stats);
for i = 1:3
    b(i).FaceColor = colors_class(i, :);
end
set(gca, 'XTickLabel', bands.names);
ylabel('Importance');
title('Band Importance', 'FontSize', 12, 'FontWeight', 'bold');
legend(classes, 'Location', 'northeast', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 9);

% --- Panel F: Key findings ---
subplot(3, 4, 11:12);
axis off;

% Find most important band for each class
[~, max_band_idx] = max(band_stats, [], 1);
most_important_bands = bands.names(max_band_idx);

% Find peak time for each class
peak_times = zeros(1, 3);
for i = 1:3
    [~, peak_idx] = max(temp_profiles(:, i));
    peak_times(i) = time_axis(peak_idx);
end

% Compute band differences
theta_idx = 2;
alpha_idx = 3;
theta_diff_coop_comp = band_stats(theta_idx, 3) - band_stats(theta_idx, 2);
alpha_diff_coop_comp = band_stats(alpha_idx, 3) - band_stats(alpha_idx, 2);

findings_text = {
    '=== Key Findings ===', ...
    '', ...
    '--- Most Important Band per Class ---', ...
    sprintf('  Single:      %s', most_important_bands{1}), ...
    sprintf('  Competition: %s', most_important_bands{2}), ...
    sprintf('  Cooperation: %s', most_important_bands{3}), ...
    '', ...
    '--- Peak Importance Time ---', ...
    sprintf('  Single:      %.0f ms', peak_times(1)), ...
    sprintf('  Competition: %.0f ms', peak_times(2)), ...
    sprintf('  Cooperation: %.0f ms', peak_times(3)), ...
    '', ...
    '--- Coop vs Comp Differences ---', ...
    sprintf('  Theta: %+.2e', theta_diff_coop_comp), ...
    sprintf('  Alpha: %+.2e', alpha_diff_coop_comp)
};

text(0.05, 0.95, findings_text, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontSize', 10, 'FontName', 'FixedWidth');
title('Key Findings', 'FontSize', 12, 'FontWeight', 'bold');

sgtitle('Grad-CAM Analysis Summary', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig7, fullfile(output_dir, 'gradcam_summary.png'));
fprintf('Saved: gradcam_summary.png\n');


%% ========================================================================
% Export Statistics
% =========================================================================

fprintf('\n--- Exporting Statistics ---\n');

% Frequency profile export
freq_profile_table = array2table([freq_axis', freq_profiles], ...
    'VariableNames', ['Frequency_Hz', classes]);
writetable(freq_profile_table, fullfile(output_dir, 'gradcam_frequency_profile.csv'));
fprintf('Saved: gradcam_frequency_profile.csv\n');

% Temporal profile export
temp_profile_table = array2table([time_axis', temp_profiles], ...
    'VariableNames', ['Time_ms', classes]);
writetable(temp_profile_table, fullfile(output_dir, 'gradcam_temporal_profile.csv'));
fprintf('Saved: gradcam_temporal_profile.csv\n');


%% ========================================================================
% Done
% =========================================================================

fprintf('\n=== Grad-CAM Analysis Complete ===\n');
fprintf('All figures saved to: %s\n', output_dir);
fprintf('\nGenerated files:\n');
fprintf('  1. gradcam_heatmap_comparison.png  - Three-class heatmaps\n');
fprintf('  2. gradcam_difference_maps.png     - Pairwise differences\n');
fprintf('  3. gradcam_frequency_profile.png   - Frequency importance\n');
fprintf('  4. gradcam_temporal_profile.png    - Temporal importance\n');
fprintf('  5. gradcam_band_statistics.png     - Band analysis\n');
fprintf('  6. gradcam_3d_surface.png          - 3D visualization\n');
fprintf('  7. gradcam_summary.png             - Publication summary\n');
fprintf('  8. gradcam_band_stats.csv          - Band statistics\n');
fprintf('  9. gradcam_frequency_profile.csv   - Frequency data\n');
fprintf(' 10. gradcam_temporal_profile.csv    - Temporal data\n');
