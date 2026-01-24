% Analyze Cross-Attention Weights Results
% This script reads attention weight data from DualEEG model analysis
% and generates visualizations:
% 1. Attention Matrix Heatmap (Full 139x139)
% 2. Diagonal Profile Plot (Time Synchrony)
% 3. Class Comparison Bar Chart
% 4. Time-lag Analysis (Off-diagonal)
% 5. Attention Distribution Histogram
%
% Data source: attention_weights/ from Python analysis pipeline

clear; clc; close all;


%% ========================================================================
% Configuration
% =========================================================================

% Get the directory of the current script
script_dir = fileparts(mfilename('fullpath'));

% Define paths
data_dir = fullfile(script_dir, '..', 'raw_result', 'dualEEG_old_eeg', 'attention_weights');
output_dir = fullfile(script_dir, '..', 'figures', 'attention_weights_matlab');

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

% Define file paths
npy_file = fullfile(data_dir, 'mean_attention_map.npy');
csv_file = fullfile(data_dir, 'mean_attention_map.csv');
summary_file = fullfile(data_dir, 'attention_diagonal_summary.csv');

% Define custom colors (consistent with project style)
colors_class = [0.55, 0.63, 0.80;   % Single - Blue
                0.99, 0.55, 0.38;   % Competition - Orange
                0.40, 0.76, 0.65];  % Cooperation - Green
classes = {'Single', 'Competition', 'Cooperation'};

fprintf('\n=== Attention Weights Analysis ===\n');

% =========================================================================
% Sequence Structure (from DualEEGTransformer architecture)
% =========================================================================
% The 139-length sequence is composed of:
%   [CLS, IBS(1..42), Spec(1..32), Temporal(1..64)]
%   Position 0:      CLS token
%   Position 1-42:   IBS tokens (6 bands x 7 features = 42)
%   Position 43-74:  Spectrogram tokens (32 channels)
%   Position 75-138: Temporal tokens (64 time steps)
%
% Temporal tokens represent compressed time from original EEG:
%   - Original: 1024 samples at 256 Hz = 4 seconds
%   - After 2x Conv1d (stride=4): 1024 -> 256 -> 64 tokens
%   - Each temporal token spans ~16 samples = ~62.5 ms

% Define sequence boundaries
SEQ_CLS_END = 1;           % Position 0
SEQ_IBS_START = 2;         % Position 1
SEQ_IBS_END = 43;          % Position 42 (1-indexed in MATLAB)
SEQ_SPEC_START = 44;       % Position 43
SEQ_SPEC_END = 75;         % Position 74
SEQ_TEMP_START = 76;       % Position 75
% SEQ_TEMP_END = seq_len;  % Position 138

fprintf('Sequence structure:\n');
fprintf('  CLS:      position 1\n');
fprintf('  IBS:      positions 2-43 (42 tokens)\n');
fprintf('  Spec:     positions 44-75 (32 tokens)\n');
fprintf('  Temporal: positions 76-%d (64 tokens)\n', 139);


%% ========================================================================
% Load Data
% =========================================================================

% Step 1: Convert .npy to .csv if needed (using Python)
if ~exist(csv_file, 'file')
    fprintf('Converting .npy to .csv using Python...\n');

    % Create Python conversion command
    py_cmd = sprintf('python -c "import numpy as np; data = np.load(r''%s''); np.savetxt(r''%s'', data, delimiter='','')"', ...
        strrep(npy_file, '\', '/'), strrep(csv_file, '\', '/'));

    [status, result] = system(py_cmd);
    if status ~= 0
        error('Failed to convert .npy file. Make sure Python and NumPy are installed.\nError: %s', result);
    end
    fprintf('Conversion successful!\n');
end

% Step 2: Load the attention matrix
if exist(csv_file, 'file')
    fprintf('Loading attention matrix from: %s\n', csv_file);
    attention_matrix = readmatrix(csv_file);
    [seq_len, ~] = size(attention_matrix);
    fprintf('Matrix size: %d x %d\n', seq_len, seq_len);
else
    error('Attention matrix file not found: %s', csv_file);
end

% Step 3: Load diagonal summary
if exist(summary_file, 'file')
    fprintf('Loading diagonal summary from: %s\n', summary_file);
    summary_table = readtable(summary_file);
    disp(summary_table);
else
    warning('Diagonal summary file not found: %s', summary_file);
    summary_table = [];
end


%% ========================================================================
% 1. Attention Matrix Heatmap
% =========================================================================

fprintf('\n--- 1. Generating Attention Matrix Heatmap ---\n');

fig1 = figure('Name', 'Attention Matrix Heatmap', ...
    'Color', 'w', 'Position', [100, 100, 800, 700]);

imagesc(attention_matrix);
colormap(hot);
colorbar;
axis square;

% Add diagonal line and sequence boundary indicators
hold on;
plot([1, seq_len], [1, seq_len], 'c--', 'LineWidth', 2);

% Add boundary lines for sequence structure
boundary_color = [0.3, 0.8, 0.3];  % Green
boundaries = [SEQ_IBS_END, SEQ_SPEC_END];  % After IBS, After Spec
for b = boundaries
    plot([b, b], [1, seq_len], '--', 'Color', boundary_color, 'LineWidth', 1.5);
    plot([1, seq_len], [b, b], '--', 'Color', boundary_color, 'LineWidth', 1.5);
end
hold off;

% Labels
title('Cross-Attention Map (Brain 1 \rightarrow Brain 2)', ...
    'FontSize', 14, 'FontWeight', 'bold');
xlabel('Brain 2 Sequence Position (Key)', 'FontSize', 12);
ylabel('Brain 1 Sequence Position (Query)', 'FontSize', 12);

% Add annotations
text(seq_len*0.02, seq_len*0.98, 'Cyan = Diagonal (time-sync)', ...
    'Color', 'c', 'FontSize', 9, 'VerticalAlignment', 'top');
text(seq_len*0.02, seq_len*0.93, 'Green = Token boundaries', ...
    'Color', boundary_color, 'FontSize', 9, 'VerticalAlignment', 'top');

% Add region labels
text(SEQ_IBS_END/2, -5, 'IBS', 'HorizontalAlignment', 'center', 'FontSize', 9);
text((SEQ_IBS_END + SEQ_SPEC_END)/2, -5, 'Spec', 'HorizontalAlignment', 'center', 'FontSize', 9);
text((SEQ_SPEC_END + seq_len)/2, -5, 'Temporal', 'HorizontalAlignment', 'center', 'FontSize', 9);

% Save
saveas(fig1, fullfile(output_dir, 'attention_heatmap.png'));
fprintf('Saved: attention_heatmap.png\n');


%% ========================================================================
% 2. Diagonal Profile Plot (Time Synchrony)
% =========================================================================

fprintf('\n--- 2. Generating Diagonal Profile Plot ---\n');

fig2 = figure('Name', 'Attention Diagonal Profile', ...
    'Color', 'w', 'Position', [100, 100, 1000, 500]);

% Extract diagonal
diagonal_values = diag(attention_matrix);
time_steps = 1:seq_len;

% Smooth for visualization
window_size = 5;
diagonal_smooth = movmean(diagonal_values, window_size);

% Plot
subplot(2, 1, 1);

% Background shading for different token types
hold on;
y_max = max(diagonal_values) * 1.1;
y_min = min(diagonal_values) * 0.9;

% IBS region (light blue)
fill([SEQ_IBS_START, SEQ_IBS_END, SEQ_IBS_END, SEQ_IBS_START], ...
     [y_min, y_min, y_max, y_max], [0.8, 0.9, 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
% Spec region (light green)
fill([SEQ_SPEC_START, SEQ_SPEC_END, SEQ_SPEC_END, SEQ_SPEC_START], ...
     [y_min, y_min, y_max, y_max], [0.8, 1.0, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
% Temporal region (light orange)
fill([SEQ_TEMP_START, seq_len, seq_len, SEQ_TEMP_START], ...
     [y_min, y_min, y_max, y_max], [1.0, 0.9, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

plot(time_steps, diagonal_values, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 0.5);
plot(time_steps, diagonal_smooth, 'Color', [0.2, 0.4, 0.8], 'LineWidth', 2);
hold off;

xlabel('Sequence Position', 'FontSize', 11);
ylabel('Attention Weight', 'FontSize', 11);
title('Diagonal Attention Profile by Token Type', 'FontSize', 13, 'FontWeight', 'bold');
legend({'IBS region', 'Spec region', 'Temporal region', 'Raw', 'Smoothed'}, ...
    'Location', 'northeast', 'FontSize', 8);
grid on;
xlim([1, seq_len]);

% Add region labels at top
text((SEQ_IBS_START + SEQ_IBS_END)/2, y_max*0.98, 'IBS', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
text((SEQ_SPEC_START + SEQ_SPEC_END)/2, y_max*0.98, 'Spec', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
text((SEQ_TEMP_START + seq_len)/2, y_max*0.98, 'Temporal', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);

% Highlight peaks with focus on temporal region
subplot(2, 1, 2);
% Find local maxima
[pks, locs] = findpeaks(diagonal_smooth, 'MinPeakProminence', 0.0005);

hold on;
% Add vertical lines at token boundaries
xline(SEQ_IBS_END, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1);
xline(SEQ_SPEC_END, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1);

plot(time_steps, diagonal_smooth, 'Color', [0.2, 0.4, 0.8], 'LineWidth', 2);

if ~isempty(pks)
    plot(locs, pks, 'rv', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    % Annotate peaks in temporal region
    for p = 1:length(locs)
        if locs(p) >= SEQ_TEMP_START
            temp_pos = locs(p) - SEQ_TEMP_START + 1;
            orig_time_ms = temp_pos * 16 / 256 * 1000;  % Convert to ms
            text(locs(p), pks(p) + 0.0005, sprintf('T%d\n(%.0fms)', temp_pos, orig_time_ms), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
end

yline(mean(diagonal_values), '--k', sprintf('Mean = %.4f', mean(diagonal_values)), ...
    'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
hold off;

xlabel('Sequence Position', 'FontSize', 11);
ylabel('Attention Weight', 'FontSize', 11);
title('Diagonal Profile with Peaks (Temporal region = original time)', 'FontSize', 13, 'FontWeight', 'bold');
if ~isempty(pks)
    legend({'IBS|Spec boundary', 'Spec|Temp boundary', 'Smoothed', 'Peaks', 'Mean'}, ...
        'Location', 'northeast', 'FontSize', 8);
else
    legend({'IBS|Spec boundary', 'Spec|Temp boundary', 'Smoothed', 'Mean'}, ...
        'Location', 'northeast', 'FontSize', 8);
end
grid on;
xlim([1, seq_len]);

% Add interpretation text
fprintf('\nPeak Analysis:\n');
if ~isempty(pks)
    for p = 1:length(pks)
        if locs(p) < SEQ_IBS_END
            region = 'IBS';
        elseif locs(p) < SEQ_SPEC_END
            region = 'Spec';
        else
            region = 'Temporal';
            temp_pos = locs(p) - SEQ_TEMP_START + 1;
            orig_time_ms = temp_pos * 16 / 256 * 1000;
            fprintf('  Peak at position %d (%s region, temporal token %d, ~%.0f ms)\n', ...
                locs(p), region, temp_pos, orig_time_ms);
        end
    end
end

% Save
saveas(fig2, fullfile(output_dir, 'attention_diagonal_profile.png'));
fprintf('Saved: attention_diagonal_profile.png\n');


%% ========================================================================
% 3. Class Comparison Bar Chart
% =========================================================================

fprintf('\n--- 3. Generating Class Comparison Bar Chart ---\n');

fig3 = figure('Name', 'Attention Class Comparison', ...
    'Color', 'w', 'Position', [100, 100, 700, 500]);

if ~isempty(summary_table)
    % Extract data from summary table
    class_names_data = summary_table.Class;
    mean_diag_values = summary_table.Mean_Diagonal_Value;
    sample_counts = summary_table.Sample_Count;

    % Reorder to match our standard order (Single, Competition, Cooperation)
    ordered_values = zeros(3, 1);
    ordered_counts = zeros(3, 1);
    for i = 1:3
        idx = find(strcmpi(class_names_data, classes{i}));
        if ~isempty(idx)
            ordered_values(i) = mean_diag_values(idx);
            ordered_counts(i) = sample_counts(idx);
        end
    end

    % Bar plot
    b = bar(ordered_values);
    b.FaceColor = 'flat';
    for i = 1:3
        b.CData(i, :) = colors_class(i, :);
    end

    % Add value labels
    hold on;
    for i = 1:3
        text(i, ordered_values(i) + 0.0001, ...
            sprintf('%.4f\n(n=%d)', ordered_values(i), ordered_counts(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    hold off;

    % Styling
    set(gca, 'XTickLabel', classes);
    xlabel('Condition', 'FontSize', 12);
    ylabel('Mean Diagonal Attention', 'FontSize', 12);
    title('Time-Synchronized Attention by Class', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;

    % Add interpretation
    [max_val, max_idx] = max(ordered_values);
    [min_val, min_idx] = min(ordered_values);
    annotation('textbox', [0.15, 0.75, 0.3, 0.15], ...
        'String', sprintf('Highest: %s\nLowest: %s', classes{max_idx}, classes{min_idx}), ...
        'FitBoxToText', 'on', 'BackgroundColor', 'w', 'EdgeColor', [0.7 0.7 0.7]);
else
    text(0.5, 0.5, 'Summary data not available', 'HorizontalAlignment', 'center');
    axis off;
end

% Save
saveas(fig3, fullfile(output_dir, 'attention_class_comparison.png'));
fprintf('Saved: attention_class_comparison.png\n');


%% ========================================================================
% 4. Time-lag Analysis (Off-diagonal)
% =========================================================================

fprintf('\n--- 4. Generating Time-lag Analysis ---\n');

fig4 = figure('Name', 'Attention Time-lag Analysis', ...
    'Color', 'w', 'Position', [100, 100, 1100, 800]);

% Compute mean attention at different time lags
max_lag = min(50, floor(seq_len / 2));  % Limit lag range
lags = -max_lag:max_lag;
lag_means = zeros(size(lags));

for i = 1:length(lags)
    lag = lags(i);
    if lag >= 0
        % Brain 1 attends to Brain 2's past (lag >= 0)
        diag_vals = diag(attention_matrix, lag);
    else
        % Brain 1 attends to Brain 2's future (lag < 0)
        diag_vals = diag(attention_matrix, lag);
    end
    lag_means(i) = mean(diag_vals);
end

% --- 4a: Lag profile plot ---
subplot(2, 2, 1:2);
bar(lags, lag_means, 'FaceColor', [0.55, 0.63, 0.80], 'EdgeColor', 'none');
hold on;
% Highlight zero lag (diagonal)
zero_idx = find(lags == 0);
bar(0, lag_means(zero_idx), 'FaceColor', [0.99, 0.55, 0.38]);
xline(0, '--r', 'LineWidth', 1.5);
hold off;

xlabel('Time Lag (Brain 2 - Brain 1)', 'FontSize', 11);
ylabel('Mean Attention Weight', 'FontSize', 11);
title('Cross-Attention by Time Lag', 'FontSize', 13, 'FontWeight', 'bold');
legend({'Off-diagonal', 'Diagonal (lag=0)', 'Zero lag'}, 'Location', 'northeast');
grid on;

% Add interpretation text
text(-max_lag*0.9, max(lag_means)*0.95, ...
    'Negative lag: B1 attends to B2''s future', ...
    'FontSize', 9, 'Color', [0.3, 0.3, 0.3]);
text(max_lag*0.3, max(lag_means)*0.95, ...
    'Positive lag: B1 attends to B2''s past', ...
    'FontSize', 9, 'Color', [0.3, 0.3, 0.3]);

% --- 4b: Asymmetry analysis ---
subplot(2, 2, 3);
% Compare positive vs negative lags
pos_lag_mean = mean(lag_means(lags > 0));
neg_lag_mean = mean(lag_means(lags < 0));
zero_lag_mean = lag_means(zero_idx);

asymmetry_data = [neg_lag_mean, zero_lag_mean, pos_lag_mean];
asymmetry_labels = {'Past (lag<0)', 'Sync (lag=0)', 'Future (lag>0)'};
asymmetry_colors = [0.4, 0.76, 0.65; 0.99, 0.55, 0.38; 0.55, 0.63, 0.80];

b2 = bar(asymmetry_data);
b2.FaceColor = 'flat';
b2.CData = asymmetry_colors;
set(gca, 'XTickLabel', asymmetry_labels);
ylabel('Mean Attention', 'FontSize', 11);
title('Temporal Asymmetry', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Add values on bars
for i = 1:3
    text(i, asymmetry_data(i) + 0.0001, sprintf('%.4f', asymmetry_data(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% --- 4c: Heatmap with lag structure ---
subplot(2, 2, 4);
% Show a zoomed portion around the diagonal
zoom_range = max(1, floor(seq_len/2) - 30):min(seq_len, floor(seq_len/2) + 30);
zoomed_matrix = attention_matrix(zoom_range, zoom_range);

imagesc(zoomed_matrix);
colormap(gca, hot);
colorbar;
axis square;

hold on;
plot([1, length(zoom_range)], [1, length(zoom_range)], 'c--', 'LineWidth', 1.5);
hold off;

title('Zoomed Center Region', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Brain 2 Time', 'FontSize', 10);
ylabel('Brain 1 Time', 'FontSize', 10);

sgtitle('Time-Lag Analysis of Cross-Attention', 'FontSize', 15, 'FontWeight', 'bold');

% Save
saveas(fig4, fullfile(output_dir, 'attention_timelag_analysis.png'));
fprintf('Saved: attention_timelag_analysis.png\n');


%% ========================================================================
% 5. Attention Distribution Histogram
% =========================================================================

fprintf('\n--- 5. Generating Attention Distribution ---\n');

fig5 = figure('Name', 'Attention Distribution', ...
    'Color', 'w', 'Position', [100, 100, 1000, 500]);

% Get all values
all_values = attention_matrix(:);
diag_values = diag(attention_matrix);
off_diag_mask = ~eye(size(attention_matrix));
off_diag_values = attention_matrix(off_diag_mask);

% --- 5a: Histogram comparison ---
subplot(1, 2, 1);
hold on;

% All values
histogram(all_values, 50, 'FaceColor', [0.7, 0.7, 0.7], 'FaceAlpha', 0.5, ...
    'EdgeColor', 'none', 'Normalization', 'probability');
% Diagonal only
histogram(diag_values, 30, 'FaceColor', [0.99, 0.55, 0.38], 'FaceAlpha', 0.7, ...
    'EdgeColor', 'none', 'Normalization', 'probability');

hold off;

xlabel('Attention Weight', 'FontSize', 11);
ylabel('Probability', 'FontSize', 11);
title('Attention Value Distribution', 'FontSize', 13, 'FontWeight', 'bold');
legend({'All values', 'Diagonal (sync)'}, 'Location', 'northeast');
grid on;

% --- 5b: Box plot comparison ---
subplot(1, 2, 2);

% Prepare data for boxplot
group_data = [diag_values; off_diag_values(1:min(1000, length(off_diag_values)))];  % Sample off-diag
group_labels = [repmat({'Diagonal'}, length(diag_values), 1); ...
                repmat({'Off-diagonal'}, min(1000, length(off_diag_values)), 1)];

boxplot(group_data, group_labels);
ylabel('Attention Weight', 'FontSize', 11);
title('Diagonal vs Off-diagonal Comparison', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% Add statistics text
stats_text = {
    sprintf('All: mean=%.4f, std=%.4f', mean(all_values), std(all_values)), ...
    sprintf('Diag: mean=%.4f, std=%.4f', mean(diag_values), std(diag_values)), ...
    sprintf('Off-diag: mean=%.4f, std=%.4f', mean(off_diag_values), std(off_diag_values))
};
annotation('textbox', [0.55, 0.15, 0.4, 0.15], ...
    'String', stats_text, 'FitBoxToText', 'on', ...
    'BackgroundColor', 'w', 'EdgeColor', [0.7 0.7 0.7], 'FontSize', 9);

% Save
saveas(fig5, fullfile(output_dir, 'attention_distribution.png'));
fprintf('Saved: attention_distribution.png\n');


%% ========================================================================
% 6. Summary Figure (Publication-ready)
% =========================================================================

fprintf('\n--- 6. Generating Summary Figure ---\n');

fig6 = figure('Name', 'Attention Summary', ...
    'Color', 'w', 'Position', [50, 50, 1400, 900]);

% --- Panel A: Heatmap ---
subplot(2, 3, 1);
imagesc(attention_matrix);
colormap(gca, hot);
colorbar;
axis square;
hold on;
plot([1, seq_len], [1, seq_len], 'c--', 'LineWidth', 1.5);
hold off;
title('(A) Attention Matrix', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Brain 2 Time');
ylabel('Brain 1 Time');
set(gca, 'FontSize', 9);

% --- Panel B: Diagonal Profile ---
subplot(2, 3, 2);
plot(time_steps, diagonal_smooth, 'Color', [0.2, 0.4, 0.8], 'LineWidth', 1.5);
hold on;
yline(mean(diagonal_values), '--k', 'LineWidth', 1);
hold off;
xlabel('Time Step');
ylabel('Attention');
title('(B) Diagonal Profile', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([1, seq_len]);
set(gca, 'FontSize', 9);

% --- Panel C: Class Comparison ---
subplot(2, 3, 3);
if ~isempty(summary_table)
    b = bar(ordered_values);
    b.FaceColor = 'flat';
    for i = 1:3
        b.CData(i, :) = colors_class(i, :);
    end
    set(gca, 'XTickLabel', classes);
    ylabel('Mean Diagonal Attn');
    title('(C) Class Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 9);
end

% --- Panel D: Time-lag ---
subplot(2, 3, 4);
bar(lags, lag_means, 'FaceColor', [0.55, 0.63, 0.80], 'EdgeColor', 'none');
hold on;
bar(0, lag_means(zero_idx), 'FaceColor', [0.99, 0.55, 0.38]);
xline(0, '--r', 'LineWidth', 1);
hold off;
xlabel('Time Lag');
ylabel('Mean Attention');
title('(D) Time-Lag Analysis', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 9);

% --- Panel E: Distribution ---
subplot(2, 3, 5);
hold on;
histogram(all_values, 40, 'FaceColor', [0.7, 0.7, 0.7], 'FaceAlpha', 0.5, ...
    'EdgeColor', 'none', 'Normalization', 'probability');
histogram(diag_values, 25, 'FaceColor', [0.99, 0.55, 0.38], 'FaceAlpha', 0.7, ...
    'EdgeColor', 'none', 'Normalization', 'probability');
hold off;
xlabel('Attention Weight');
ylabel('Probability');
title('(E) Distribution', 'FontSize', 12, 'FontWeight', 'bold');
legend({'All', 'Diagonal'}, 'Location', 'northeast', 'FontSize', 8);
grid on;
set(gca, 'FontSize', 9);

% --- Panel F: Statistics Summary ---
subplot(2, 3, 6);
axis off;

% Compute key statistics
diag_off_ratio = mean(diag_values) / mean(off_diag_values);
peak_time = find(diagonal_smooth == max(diagonal_smooth), 1);

summary_text = {
    '=== Key Statistics ===', ...
    '', ...
    sprintf('Matrix Size: %d x %d', seq_len, seq_len), ...
    '', ...
    '--- Attention Values ---', ...
    sprintf('  Global Mean: %.5f', mean(all_values)), ...
    sprintf('  Diagonal Mean: %.5f', mean(diag_values)), ...
    sprintf('  Off-diag Mean: %.5f', mean(off_diag_values)), ...
    sprintf('  Diag/Off-diag Ratio: %.3f', diag_off_ratio), ...
    '', ...
    '--- Temporal ---', ...
    sprintf('  Peak Time Step: %d', peak_time), ...
    sprintf('  Sync > Past: %s', string(zero_lag_mean > pos_lag_mean)), ...
    sprintf('  Sync > Future: %s', string(zero_lag_mean > neg_lag_mean))
};

text(0.05, 0.95, summary_text, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontSize', 10, 'FontName', 'FixedWidth');
title('(F) Summary', 'FontSize', 12, 'FontWeight', 'bold');

sgtitle('Cross-Attention Analysis Summary', 'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig6, fullfile(output_dir, 'attention_summary.png'));
fprintf('Saved: attention_summary.png\n');


%% ========================================================================
% Export Statistics
% =========================================================================

fprintf('\n--- Exporting Statistics ---\n');

% Create statistics table
stats_data = {
    'Matrix_Size', sprintf('%dx%d', seq_len, seq_len);
    'Global_Mean', sprintf('%.6f', mean(all_values));
    'Global_Std', sprintf('%.6f', std(all_values));
    'Diagonal_Mean', sprintf('%.6f', mean(diag_values));
    'Diagonal_Std', sprintf('%.6f', std(diag_values));
    'OffDiag_Mean', sprintf('%.6f', mean(off_diag_values));
    'OffDiag_Std', sprintf('%.6f', std(off_diag_values));
    'Diag_OffDiag_Ratio', sprintf('%.4f', diag_off_ratio);
    'Peak_TimeStep', sprintf('%d', peak_time);
    'Neg_Lag_Mean', sprintf('%.6f', neg_lag_mean);
    'Zero_Lag_Mean', sprintf('%.6f', zero_lag_mean);
    'Pos_Lag_Mean', sprintf('%.6f', pos_lag_mean);
};

stats_table = cell2table(stats_data, 'VariableNames', {'Metric', 'Value'});
writetable(stats_table, fullfile(output_dir, 'attention_statistics.csv'));
fprintf('Saved: attention_statistics.csv\n');

% Export lag profile
lag_table = table(lags', lag_means', 'VariableNames', {'Lag', 'Mean_Attention'});
writetable(lag_table, fullfile(output_dir, 'attention_lag_profile.csv'));
fprintf('Saved: attention_lag_profile.csv\n');


%% ========================================================================
% Done
% =========================================================================

fprintf('\n=== Attention Weights Analysis Complete ===\n');
fprintf('All figures saved to: %s\n', output_dir);
fprintf('\nGenerated files:\n');
fprintf('  1. attention_heatmap.png         - Full attention matrix\n');
fprintf('  2. attention_diagonal_profile.png - Time synchrony profile\n');
fprintf('  3. attention_class_comparison.png - Class comparison\n');
fprintf('  4. attention_timelag_analysis.png - Time-lag analysis\n');
fprintf('  5. attention_distribution.png     - Value distribution\n');
fprintf('  6. attention_summary.png          - Publication summary\n');
fprintf('  7. attention_statistics.csv       - Key statistics\n');
fprintf('  8. attention_lag_profile.csv      - Lag profile data\n');
