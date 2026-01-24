% Analyze EEG Results
% This script reads EEG analysis data and generates visualizations:
% 1.1 Boxplot by Pair ID (Sorted) - Entropy Analysis
% 1.2 Raincloud Plot by Condition - Entropy Analysis
% 1.3 Topoplot by Condition - Entropy Analysis
% 2.1 Frequency Sensitivity Analysis - DualEEG Model

clear; clc; close all;


% Get the directory of the current script
script_dir = fileparts(mfilename('fullpath'));

% Define file paths relative to script location
% Entropy analysis paths
data_path_entropy = fullfile(script_dir, '..', 'raw_result', 'entropy_analysis', 'eeg_entropy_raw_corrected.csv');
output_path_boxplot = fullfile(script_dir, '..', 'figures', 'entropy_analysis_matlab', 'eeg_entropy_boxplot_sorted.png');
output_path_raincloud = fullfile(script_dir, '..', 'figures', 'entropy_analysis_matlab', 'eeg_entropy_raincloud.png');
output_path_topoplot = fullfile(script_dir, '..', 'figures', 'entropy_analysis_matlab', 'eeg_entropy_topoplot.png');

% Frequency sensitivity paths
data_path_freq = fullfile(script_dir, '..', 'raw_result', 'dualEEG_old_eeg', 'frequency_sensitivity', 'frequency_sensitivity.csv');
output_path_freq = fullfile(script_dir, '..', 'figures', 'frequency_sensitivity_matlab', 'freq_sensitivity_grouped_bar.png');

% Verify entropy data exists
if ~exist(data_path_entropy, 'file')
    warning('Entropy data file not found: %s. Skipping entropy analysis.', data_path_entropy);
    run_entropy_analysis = false;
else
    run_entropy_analysis = true;
end

% Verify frequency sensitivity data exists
if ~exist(data_path_freq, 'file')
    warning('Frequency sensitivity data file not found: %s. Skipping frequency analysis.', data_path_freq);
    run_freq_analysis = false;
else
    run_freq_analysis = true;
end

% Load entropy data if available
if run_entropy_analysis
    data = readtable(data_path_entropy);
end

%% ========================================================================
% ENTROPY ANALYSIS (1.1 - 1.3)
% =========================================================================

if run_entropy_analysis

% Ensure pair_id is treated as categorical for correct ordering
if isnumeric(data.pair_id)
    data.pair_id = string(data.pair_id);
end
data.pair_id = categorical(data.pair_id);

% Define custom colors and order
conditions_order = {'Single', 'Competition', 'Cooperation'};
custom_colors = [0.55, 0.63, 0.80;   % Single
                 0.99, 0.55, 0.38;   % Competition
                 0.40, 0.76, 0.65];  % Cooperation

% Ensure condition is categorical and reordered to match colors
if ~iscategorical(data.condition)
    data.condition = categorical(data.condition);
end
data.condition = reordercats(data.condition, conditions_order);


% --- Calculate Statistics for Sorting (for Boxplot) ---
try
    % Step 1: Mean per Pair per Condition
    pairCondStats = grpstats(data, {'pair_id', 'condition'}, 'mean', 'DataVars', 'mean_entropy');
    
    % Step 2: Mean per Pair (averaging the condition means)
    pairStats = grpstats(pairCondStats, 'pair_id', 'mean', 'DataVars', 'mean_mean_entropy');
    
    % Sort by the calculated mean
    pairStats = sortrows(pairStats, 'mean_mean_mean_entropy', 'ascend');
    
    sortedPairs = pairStats.pair_id;
    overallMeans = pairStats.mean_mean_mean_entropy;
catch ME
    warning('Complex grouping failed, falling back to simple mean per pair. Error: %s', ME.message);
    % Fallback: Simple mean per pair
    pairStats = grpstats(data, 'pair_id', 'mean', 'DataVars', 'mean_entropy');
    pairStats = sortrows(pairStats, 'mean_mean_entropy', 'ascend');
    sortedPairs = pairStats.pair_id;
    overallMeans = pairStats.mean_mean_entropy;
end

% Apply sorting to the main data
data.pair_id = reordercats(data.pair_id, string(sortedPairs));


% --- 1.1 Visualization: EEG Entropy by Pair ID (Sorted by Mean, Horizontal) ---
disp('Generating EEG Boxplot (Sorted)...');
figure('Name', 'EEG Entropy by Pair ID', 'Color', 'w', 'Position', [100, 100, 800, 1000]);

% Set color order for the axes
colororder(custom_colors);

% Main Boxchart
boxchart(data.pair_id, data.mean_entropy, ...
    'GroupByColor', data.condition, ...
    'Orientation', 'horizontal');
hold on;

% Plot Overall Means
plot(overallMeans, 1:length(sortedPairs), '-dk', ...
    'LineWidth', 1.5, ...
    'MarkerFaceColor', 'w', 'MarkerSize', 6, ...
    'DisplayName', 'Mean (Avg of Conditions)');

% Styling
title('EEG Mean Entropy Distribution by Pair ID (Sorted)');
ylabel('Pair ID');
xlabel('Mean Entropy');
legend('Location', 'northeastoutside');
grid on;

hold off;

% Save the figure
saveas(gcf, output_path_boxplot);
disp(['Boxplot saved to: ', output_path_boxplot]);


% --- 1.2 Visualization: Raincloud Plot (EEG) ---
disp('Generating EEG Raincloud Plot (Horizontal)...');
figure('Name', 'EEG Entropy Raincloud', 'Color', 'w', 'Position', [100, 100, 800, 600]);

% Add raincloudplots to path
addpath(genpath(fullfile(script_dir, 'raincloudplots')));

% Prepare data (3x1 Cell Array) matching the order in conditions_order
data_eeg = cell(3, 1);
data_eeg{1, 1} = data.mean_entropy(data.condition == 'Single');
data_eeg{2, 1} = data.mean_entropy(data.condition == 'Competition');
data_eeg{3, 1} = data.mean_entropy(data.condition == 'Cooperation');

% Plot Horizontal (param=1)
% Use a neutral color for the initial call, will override below
h_eeg = rm_raincloud(data_eeg, [0.5 0.5 0.5], 1, 'ks');

% Apply distinct colors manually
for i = 1:3
    h_eeg.p{i, 1}.FaceColor = custom_colors(i, :);
    h_eeg.p{i, 1}.FaceVertexCData = custom_colors(i, :);
    h_eeg.s{i, 1}.MarkerFaceColor = custom_colors(i, :);
    h_eeg.s{i, 1}.MarkerEdgeColor = 'none';
    h_eeg.m(i, 1).MarkerFaceColor = custom_colors(i, :);
end

title('EEG Mean Entropy Distribution');
xlabel('Mean Entropy');
ylabel('Condition');
set(gca, 'YTick', 1:3);
set(gca, 'YTickLabel', conditions_order);
% grid on;

saveas(gcf, output_path_raincloud);
disp(['Raincloud plot saved to: ', output_path_raincloud]);


% --- 1.3 Visualization: EEG Entropy Topoplot ---
disp('Generating EEG Topoplot...');
figure('Name', 'EEG Entropy Topoplot', 'Color', 'w', 'Position', [100, 100, 1200, 400]);

% 1. Identify Channel Data
% Columns 6 to end are channels (Fp1, Fp2... based on inspection)
% Verify this by checking if column 6 is indeed a channel (usually Fp1 or similar)
% Using column indices 6:end
channel_names = data.Properties.VariableNames(6:end);
channel_data = data{:, 6:end};

% 2. Calculate Means per Condition
mean_maps = zeros(length(channel_names), 3);
for i = 1:3
    cond_name = conditions_order{i};
    cond_idx = data.condition == cond_name;
    % Calculate mean across all trials for this condition
    mean_maps(:, i) = mean(channel_data(cond_idx, :), 1)';
end

% 3. Load Channel Locations
try
    % Attempt to read standard locations. 
    % If 'readlocs' is in path (EEGLAB), this should work.
    % We use 'Standard-10-20-Cap81.ced' which is standard in EEGLAB.
    % If strict path is needed, user might need to adjust.
    std_locs = readlocs('Standard-10-20-Cap81.ced');
    
    % Match our channels to standard locations
    chanlocs = std_locs(1); % Initialize with first to set structure
    chanlocs(1:length(channel_names)) = std_locs(1); % Pre-allocate
    
    for i = 1:length(channel_names)
        idx = find(strcmpi({std_locs.labels}, channel_names{i}));
        if ~isempty(idx)
            chanlocs(i) = std_locs(idx);
        else
            warning('Channel %s not found in standard locations.', channel_names{i});
        end
    end
    
    % 4. Plotting
    % Determine global color limits for comparison
    clim_min = min(mean_maps(:));
    clim_max = max(mean_maps(:));
    
    % Create custom gradient colormap from user colors
    % Order: Single (Blue) -> Cooperation (Green) -> Competition (Orange)
    % Assumption: Low Entropy -> Blue, High Entropy -> Orange
    c_single = [0.55, 0.63, 0.80];
    c_coop   = [1.00, 1.00, 1.00];
    c_comp   = [0.99, 0.55, 0.38];
    
    % Interpolate to create a 256-color map
    xp = [1, 128, 256];
    yp = [c_single; c_coop; c_comp];
    custom_map = interp1(xp, yp, 1:256);
    
    for i = 1:3
        sub_h = subplot(1, 3, i);
        
        % Plot topoplot
        topoplot(mean_maps(:, i), chanlocs, ...
            'maplimits', [clim_min, clim_max], ...
            'style', 'map', ...
            'electrodes', 'on', ...
            'headrad', 'rim', ...
            'shading', 'interp');
            
        title(conditions_order{i});
        
        % Force colormap for this specific subplot axis
        colormap(sub_h, custom_map);
    end
    
    % Add common colorbar
    % Position it to the right of the last subplot
    hp = get(sub_h, 'Position');
    colorbar('Position', [hp(1)+hp(3)+0.01  hp(2)  0.02  hp(4)]);
    
    % Save
    saveas(gcf, output_path_topoplot);
    disp(['Topoplot saved to: ', output_path_topoplot]);
    
catch ME
    disp('Error generating Topoplot. Make sure EEGLAB is in your path and readlocs is available.');
    disp(['Error Details: ', ME.message]);
end

end  % end of run_entropy_analysis


%% ========================================================================
% FREQUENCY SENSITIVITY ANALYSIS (1.4)
% =========================================================================

if run_freq_analysis

% --- 2.1 Visualization: Frequency Sensitivity (Grouped Bar Chart) ---
disp('Generating Frequency Sensitivity Analysis...');

% Load frequency sensitivity data
freq_data = readtable(data_path_freq);

% Display raw data
disp('=== Frequency Sensitivity Data ===');
disp(freq_data);

% Extract variables
bands = freq_data.Band;
masked_acc = freq_data.Masked_Accuracy;
masked_f1 = freq_data.Masked_F1;
acc_drop = freq_data.Accuracy_Drop * 100;  % Convert to percentage
f1_drop = freq_data.F1_Drop * 100;

% Find baseline (gamma has 0 drop)
baseline_idx = find(acc_drop == 0);
if ~isempty(baseline_idx)
    baseline_acc = masked_acc(baseline_idx);
    baseline_f1 = masked_f1(baseline_idx);
    fprintf('Baseline (no masking): Accuracy = %.4f, F1 = %.4f\n', baseline_acc, baseline_f1);
end

num_bands = length(bands);

% Create output directory if needed
output_dir_freq = fileparts(output_path_freq);
if ~exist(output_dir_freq, 'dir')
    mkdir(output_dir_freq);
end

% Define colors for this chart
color_accuracy = [0.55, 0.63, 0.80];   % Blue-ish
color_f1 = [0.99, 0.55, 0.38];         % Orange-ish

% Create figure
figure('Name', 'Frequency Sensitivity - Grouped Bar', ...
    'Color', 'w', 'Position', [100, 100, 900, 500]);

% Create categorical x-axis preserving order
X = categorical(bands, bands);
bar_data = [acc_drop, f1_drop];

% Plot grouped bars
b = bar(X, bar_data, 'grouped');
b(1).FaceColor = color_accuracy;
b(2).FaceColor = color_f1;

% Add zero line
hold on;
yline(0, '--k', 'LineWidth', 1.2);
hold off;

% Labels and styling
ylabel('Performance Drop (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Frequency Band', 'FontSize', 12, 'FontWeight', 'bold');
title('Frequency Sensitivity: Performance Drop when Band Masked', ...
    'FontSize', 14, 'FontWeight', 'bold');
legend({'Accuracy Drop', 'F1 Drop'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 11);

% Add value labels on bars
for i = 1:2
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = arrayfun(@(x) sprintf('%.2f', x), bar_data(:,i), 'UniformOutput', false);
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 8);
end

% Save
saveas(gcf, output_path_freq);
disp(['Frequency sensitivity plot saved to: ', output_path_freq]);

end  % end of run_freq_analysis


%% ========================================================================
% 3.1 IBS CONNECTIVITY ANALYSIS - ROI-based Inter-Brain Synchrony
% =========================================================================

% Define IBS connectivity paths
data_dir_ibs = fullfile(script_dir, '..', 'raw_result', 'dualEEG_old_eeg', 'ibs_connectivity');
mean_by_class_dir = fullfile(data_dir_ibs, 'ibs_mean_by_class');
output_dir_ibs = fullfile(script_dir, '..', 'figures', 'ibs_connectivity_matlab');

% Check if IBS data exists
if ~exist(data_dir_ibs, 'dir')
    warning('IBS connectivity data directory not found: %s. Skipping IBS analysis.', data_dir_ibs);
    run_ibs_analysis = false;
else
    run_ibs_analysis = true;
end

if run_ibs_analysis

% Create output directory
if ~exist(output_dir_ibs, 'dir')
    mkdir(output_dir_ibs);
end

% Define analysis parameters
classes = {'Single', 'Competition', 'Cooperation'};
key_band = 'theta';
key_feature = 'PLV';

% Define custom colors for classes
colors_class = [0.55, 0.63, 0.80;   % Single - Blue
                0.99, 0.55, 0.38;   % Competition - Orange
                0.40, 0.76, 0.65];  % Cooperation - Green

% Load channel names
channel_file = fullfile(data_dir_ibs, 'channel_names.csv');
if exist(channel_file, 'file')
    channel_table = readtable(channel_file);
    channel_names_ibs = channel_table.Channel_Name;
    num_channels = length(channel_names_ibs);
    fprintf('Loaded %d channel names\n', num_channels);
else
    % Fallback to default 32-channel names
    channel_names_ibs = {'Fp1','Fz','F3','F7','FT9','FC5','FC1','C3',...
                     'T7','TP9','CP5','CP1','PZ','P3','P7','O1',...
                     'OZ','O2','P4','P8','TP10','CP6','CP2','CZ',...
                     'C4','T8','FT10','FC6','FC2','F4','F8','FP2'}';
    num_channels = 32;
    warning('Channel names file not found, using defaults');
end

% Define ROI (Region of Interest) groupings
roi_names = {'Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal'};
roi_channels = {
    {'Fp1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6'}, ... % Frontal
    {'C3', 'C4', 'CZ', 'CP1', 'CP2', 'CP5', 'CP6'}, ...                            % Central
    {'PZ', 'P3', 'P4', 'P7', 'P8'}, ...                                             % Parietal
    {'O1', 'O2', 'OZ'}, ...                                                         % Occipital
    {'T7', 'T8', 'TP9', 'TP10', 'FT9', 'FT10'} ...                                  % Temporal
};

fprintf('\n=== 3.1 IBS Connectivity ROI Analysis ===\n');
fprintf('Key Band: %s, Key Feature: %s\n', key_band, key_feature);

% Load matrices for each class
matrices = cell(1, 3);
for i = 1:3
    filepath = fullfile(mean_by_class_dir, ...
        sprintf('%s_%s_%s.csv', classes{i}, key_band, key_feature));
    if exist(filepath, 'file')
        matrices{i} = readmatrix(filepath);
        fprintf('Loaded matrix for %s: %dx%d\n', classes{i}, size(matrices{i}, 1), size(matrices{i}, 2));
    else
        warning('Could not load matrix for %s: %s', classes{i}, filepath);
        matrices{i} = [];
    end
end

% Check if all matrices loaded successfully
if any(cellfun(@isempty, matrices))
    warning('Some IBS matrices could not be loaded. Skipping ROI analysis.');
else

% Get channel indices for each ROI
num_rois = length(roi_names);
roi_indices = cell(1, num_rois);
for r = 1:num_rois
    roi_idx = [];
    for ch = 1:length(roi_channels{r})
        idx = find(strcmpi(channel_names_ibs, roi_channels{r}{ch}));
        if ~isempty(idx)
            roi_idx = [roi_idx, idx];
        end
    end
    roi_indices{r} = roi_idx;
end

% Compute ROI-to-ROI connectivity for each class
roi_connectivity = zeros(num_rois, num_rois, 3);
for class_idx = 1:3
    mat = matrices{class_idx};
    for r1 = 1:num_rois
        for r2 = 1:num_rois
            idx1 = roi_indices{r1};
            idx2 = roi_indices{r2};
            if ~isempty(idx1) && ~isempty(idx2)
                sub_mat = mat(idx1, idx2);
                roi_connectivity(r1, r2, class_idx) = mean(sub_mat(:));
            end
        end
    end
end

% --- Figure: ROI-based Inter-Brain Synchrony Analysis ---
disp('Generating ROI-based IBS Analysis figure...');

fig_roi = figure('Name', 'IBS ROI Analysis', ...
    'Color', 'w', 'Position', [100, 100, 1200, 800]);

% Create custom gradient colormap
% Order: Single (Blue) -> Coop (White) -> Competition (Orange)
c_single = [0.55, 0.63, 0.80];
c_coop   = [1.00, 1.00, 1.00];
c_comp   = [0.99, 0.55, 0.38];

% Interpolate to create a 256-color map
xp = [1, 128, 256];
yp = [c_single; c_coop; c_comp];
custom_gradient_map = interp1(xp, yp, 1:256);

% Find global color limits for fair comparison
all_roi_vals = roi_connectivity(:);
clim_min_roi = min(all_roi_vals);
clim_max_roi = max(all_roi_vals);

% --- Panel A: ROI Connectivity Matrices (3 classes) ---
for class_idx = 1:3
    subplot(2, 3, class_idx);
    imagesc(roi_connectivity(:, :, class_idx));
    colormap(gca, custom_gradient_map);
    clim([clim_min_roi, clim_max_roi]);
    axis square;
    colorbar;

    title(classes{class_idx}, 'FontSize', 12, 'FontWeight', 'bold', ...
        'Color', colors_class(class_idx, :));
    set(gca, 'XTick', 1:num_rois, 'XTickLabel', roi_names, ...
        'YTick', 1:num_rois, 'YTickLabel', roi_names);
    xtickangle(45);
    xlabel('Brain 2 ROI');
    ylabel('Brain 1 ROI');
end

% --- Panel B: ROI Bar Comparison (Same-region connectivity) ---
subplot(2, 3, 4:6);

% Extract diagonal (same-region connectivity)
same_region = zeros(3, num_rois);
for class_idx = 1:3
    mat = roi_connectivity(:, :, class_idx);
    same_region(class_idx, :) = diag(mat);
end

% Bar plot for same-region connectivity
bar_data_roi = same_region';  % (num_rois x 3)
b_roi = bar(bar_data_roi);
for i = 1:3
    b_roi(i).FaceColor = colors_class(i, :);
end

set(gca, 'XTickLabel', roi_names);
xlabel('Brain Region', 'FontSize', 12);
ylabel(sprintf('Mean %s', key_feature), 'FontSize', 12);
title('Same-Region Inter-Brain Connectivity by Class', 'FontSize', 14, 'FontWeight', 'bold');
legend(classes, 'Location', 'northeast');
grid on;

sgtitle(sprintf('ROI-based Inter-Brain Synchrony Analysis (%s %s)', key_band, key_feature), ...
    'FontSize', 16, 'FontWeight', 'bold');

% Save figure
output_path_roi = fullfile(output_dir_ibs, 'ibs_roi_analysis.png');
saveas(fig_roi, output_path_roi);
fprintf('Saved: %s\n', output_path_roi);

% Save ROI statistics to CSV
roi_stats_table = array2table(same_region, 'VariableNames', roi_names, 'RowNames', classes);
roi_stats_path = fullfile(output_dir_ibs, 'ibs_roi_stats.csv');
writetable(roi_stats_table, roi_stats_path, 'WriteRowNames', true);
fprintf('Saved: %s\n', roi_stats_path);

end  % end of matrices check

end  % end of run_ibs_analysis


%% ========================================================================
% 4.1 ATTENTION WEIGHTS ANALYSIS - Cross-Attention Matrix Heatmap
% =========================================================================

% Define attention weights paths
data_dir_attn = fullfile(script_dir, '..', 'raw_result', 'dualEEG_old_eeg', 'attention_weights');
output_dir_attn = fullfile(script_dir, '..', 'figures', 'attention_weights_matlab');

% Define file paths
npy_file_attn = fullfile(data_dir_attn, 'mean_attention_map.npy');
csv_file_attn = fullfile(data_dir_attn, 'mean_attention_map.csv');

% Check if attention data exists
if ~exist(data_dir_attn, 'dir')
    warning('Attention weights data directory not found: %s. Skipping attention analysis.', data_dir_attn);
    run_attn_analysis = false;
else
    run_attn_analysis = true;
end

if run_attn_analysis

% Create output directory
if ~exist(output_dir_attn, 'dir')
    mkdir(output_dir_attn);
end

fprintf('\n=== 4.1 Attention Weights Analysis ===\n');

% =========================================================================
% Sequence Structure (from DualEEGTransformer architecture)
% =========================================================================
% The 139-length sequence is composed of:
%   [CLS, IBS(1..42), Spec(1..32), Temporal(1..64)]
%   Position 0:      CLS token
%   Position 1-42:   IBS tokens (6 bands x 7 features = 42)
%   Position 43-74:  Spectrogram tokens (32 channels)
%   Position 75-138: Temporal tokens (64 time steps)

% Define sequence boundaries (MATLAB 1-indexed)
SEQ_CLS_END = 1;           % Position 0
SEQ_IBS_START = 2;         % Position 1
SEQ_IBS_END = 43;          % Position 42 (1-indexed in MATLAB)
SEQ_SPEC_START = 44;       % Position 43
SEQ_SPEC_END = 75;         % Position 74
SEQ_TEMP_START = 76;       % Position 75

% Convert .npy to .csv if needed (using Python)
if ~exist(csv_file_attn, 'file')
    if exist(npy_file_attn, 'file')
        fprintf('Converting .npy to .csv using Python...\n');
        py_cmd = sprintf('python -c "import numpy as np; data = np.load(r''%s''); np.savetxt(r''%s'', data, delimiter='','')"', ...
            strrep(npy_file_attn, '\', '/'), strrep(csv_file_attn, '\', '/'));
        [status, result] = system(py_cmd);
        if status ~= 0
            warning('Failed to convert .npy file: %s', result);
        else
            fprintf('Conversion successful!\n');
        end
    end
end

% Load the attention matrix
if exist(csv_file_attn, 'file')
    fprintf('Loading attention matrix from: %s\n', csv_file_attn);
    attention_matrix = readmatrix(csv_file_attn);
    [seq_len, ~] = size(attention_matrix);
    fprintf('Matrix size: %d x %d\n', seq_len, seq_len);

    % --- Figure: Attention Matrix Heatmap ---
    disp('Generating Attention Matrix Heatmap...');

    fig_attn = figure('Name', 'Attention Matrix Heatmap', ...
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
    output_path_attn = fullfile(output_dir_attn, 'attention_heatmap.png');
    saveas(fig_attn, output_path_attn);
    fprintf('Saved: %s\n', output_path_attn);

else
    warning('Attention matrix file not found: %s', csv_file_attn);
end

end  % end of run_attn_analysis


%% ========================================================================
% 5.1 GRAD-CAM ANALYSIS - Time-Frequency Importance Heatmap
% =========================================================================

% Define Grad-CAM paths
data_dir_gradcam = fullfile(script_dir, '..', 'raw_result', 'dualEEG_old_eeg', 'gradcam');
output_dir_gradcam = fullfile(script_dir, '..', 'figures', 'gradcam_matlab');

% Check if Grad-CAM data exists
if ~exist(data_dir_gradcam, 'dir')
    warning('Grad-CAM data directory not found: %s. Skipping Grad-CAM analysis.', data_dir_gradcam);
    run_gradcam_analysis = false;
else
    run_gradcam_analysis = true;
end

if run_gradcam_analysis

% Create output directory
if ~exist(output_dir_gradcam, 'dir')
    mkdir(output_dir_gradcam);
end

fprintf('\n=== 5.1 Grad-CAM Analysis ===\n');

% Class names and colors
classes_gradcam = {'Single', 'Competition', 'Cooperation'};
colors_class_gradcam = [0.55, 0.63, 0.80;   % Single - Blue
                        0.99, 0.55, 0.38;   % Competition - Orange
                        0.40, 0.76, 0.65];  % Cooperation - Green

% Spectrogram parameters (from model config)
sampling_rate_gc = 256;  % Hz
n_fft_gc = 128;
hop_length_gc = 64;
freq_bins_gc = 64;
time_steps_gc = 64;

% Frequency resolution
freq_resolution_gc = sampling_rate_gc / n_fft_gc;  % Hz per bin

% Time resolution
time_resolution_gc = hop_length_gc / sampling_rate_gc;  % seconds per step

% Define frequency bands (for boundary lines)
bands_hz = {[0.5, 4], [4, 8], [8, 13], [13, 30], [30, 45]};  % Delta, Theta, Alpha, Beta, Gamma

% Create frequency and time axes
freq_axis_gc = (0:freq_bins_gc-1) * freq_resolution_gc;  % Hz
time_axis_gc = (0:time_steps_gc-1) * time_resolution_gc * 1000;  % ms

fprintf('Spectrogram parameters:\n');
fprintf('  Frequency resolution: %.2f Hz/bin\n', freq_resolution_gc);
fprintf('  Time resolution: %.3f s/step\n', time_resolution_gc);

% Load Grad-CAM data
fprintf('Loading Grad-CAM data...\n');
gradcam_data = cell(1, 3);
gradcam_loaded = true;

for i = 1:3
    filepath = fullfile(data_dir_gradcam, sprintf('gradcam_%s.csv', classes_gradcam{i}));
    if exist(filepath, 'file')
        gradcam_data{i} = readmatrix(filepath);
        fprintf('  Loaded %s: %dx%d\n', classes_gradcam{i}, size(gradcam_data{i}, 1), size(gradcam_data{i}, 2));
    else
        warning('File not found: %s', filepath);
        gradcam_loaded = false;
    end
end

if gradcam_loaded
    % --- Figure: Grad-CAM Heatmap Comparison ---
    disp('Generating Grad-CAM Heatmap Comparison...');

    fig_gradcam = figure('Name', 'Grad-CAM Heatmap Comparison', ...
        'Color', 'w', 'Position', [50, 100, 1400, 450]);

    % Find global color limits
    all_vals_gc = [gradcam_data{1}(:); gradcam_data{2}(:); gradcam_data{3}(:)];
    clim_val_gc = [min(all_vals_gc), max(all_vals_gc)];

    for i = 1:3
        subplot(1, 3, i);

        % Plot heatmap
        imagesc(time_axis_gc, freq_axis_gc, gradcam_data{i});
        set(gca, 'YDir', 'normal');  % Frequency increases upward
        colormap(gca, jet);
        clim(clim_val_gc);
        colorbar;

        % Add frequency band boundaries
        hold on;
        for b = 1:5
            yline(bands_hz{b}(2), '--w', 'LineWidth', 1, 'Alpha', 0.7);
        end
        hold off;

        % Labels
        title(classes_gradcam{i}, 'FontSize', 14, 'FontWeight', 'bold', 'Color', colors_class_gradcam(i,:));
        xlabel('Time (ms)', 'FontSize', 11);
        ylabel('Frequency (Hz)', 'FontSize', 11);
        ylim([0, 50]);  % Focus on 0-50 Hz (EEG relevant range)
    end

    sgtitle('Grad-CAM: Time-Frequency Importance by Class', 'FontSize', 16, 'FontWeight', 'bold');

    % Save
    output_path_gradcam = fullfile(output_dir_gradcam, 'gradcam_heatmap_comparison.png');
    saveas(fig_gradcam, output_path_gradcam);
    fprintf('Saved: %s\n', output_path_gradcam);

else
    warning('Some Grad-CAM data could not be loaded. Skipping visualization.');
end

end  % end of run_gradcam_analysis


%% ========================================================================
% DONE
% =========================================================================
disp('=== EEG Analysis Complete ===');
