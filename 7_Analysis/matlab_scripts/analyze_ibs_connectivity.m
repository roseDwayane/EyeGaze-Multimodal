% Analyze IBS (Inter-Brain Synchrony) Connectivity Results
% This script reads IBS connectivity matrices from DualEEG model analysis
% and generates visualizations:
% 1. Connectivity Matrix Heatmap (3 classes comparison)
% 2. Difference Matrix Heatmap (Cooperation vs Competition)
% 3. Circular Connectivity Graph
% 4. ROI-based Statistics (Brain region analysis)
% 5. Multi-band Comparison
%
% Data source: ibs_connectivity/ from Python analysis pipeline

clear; clc; close all;


%% ========================================================================
% Configuration
% =========================================================================

% Get the directory of the current script
script_dir = fileparts(mfilename('fullpath'));

% Define paths
data_dir = fullfile(script_dir, '..', 'raw_result', 'dualEEG_old_eeg', 'ibs_connectivity');
mean_by_class_dir = fullfile(data_dir, 'ibs_mean_by_class');
diff_dir = fullfile(data_dir, 'ibs_difference_coop_vs_comp');
output_dir = fullfile(script_dir, '..', 'figures', 'ibs_connectivity_matlab');

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

% Verify data exists
if ~exist(data_dir, 'dir')
    error('IBS connectivity data directory not found: %s', data_dir);
end

% Define analysis parameters
classes = {'Single', 'Competition', 'Cooperation'};
bands = {'broadband', 'delta', 'theta', 'alpha', 'beta', 'gamma'};
features = {'PLV', 'PLI', 'wPLI', 'Coherence', 'Power_Corr', 'Phase_Diff', 'Time_Corr'};

% Key combinations to visualize (most relevant for hyperscanning)
key_band = 'theta'; % delta, theta, alpha, beta, gamma, broadband
key_feature = 'PLV'; % Coherence, Phase_Diff, PLI, PLV, Power_Corr, Time_Corr, wPLI

% Define custom colors (consistent with project style)
colors_class = [0.55, 0.63, 0.80;   % Single - Blue
                0.99, 0.55, 0.38;   % Competition - Orange
                0.40, 0.76, 0.65];  % Cooperation - Green

% Load channel names
channel_file = fullfile(data_dir, 'channel_names.csv');
if exist(channel_file, 'file')
    channel_table = readtable(channel_file);
    channel_names = channel_table.Channel_Name;
    num_channels = length(channel_names);
    fprintf('Loaded %d channel names\n', num_channels);
else
    % Fallback to default 32-channel names
    channel_names = {'Fp1','Fz','F3','F7','FT9','FC5','FC1','C3',...
                     'T7','TP9','CP5','CP1','PZ','P3','P7','O1',...
                     'OZ','O2','P4','P8','TP10','CP6','CP2','CZ',...
                     'C4','T8','FT10','FC6','FC2','F4','F8','FP2'}';
    num_channels = 32;
    warning('Channel names file not found, using defaults');
end

% Define ROI (Region of Interest) groupings
% Based on standard 10-20 system regions
roi_names = {'Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal'};
roi_channels = {
    {'Fp1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6'}, ... % Frontal
    {'C3', 'C4', 'CZ', 'CP1', 'CP2', 'CP5', 'CP6'}, ...                            % Central
    {'PZ', 'P3', 'P4', 'P7', 'P8'}, ...                                             % Parietal
    {'O1', 'O2', 'OZ'}, ...                                                         % Occipital
    {'T7', 'T8', 'TP9', 'TP10', 'FT9', 'FT10'} ...                                  % Temporal
};

fprintf('\n=== IBS Connectivity Analysis ===\n');
fprintf('Key Band: %s, Key Feature: %s\n', key_band, key_feature);





%% ========================================================================
% 1. Connectivity Matrix Heatmap (3 Classes Comparison)
% =========================================================================

fprintf('\n--- 1. Generating Connectivity Matrix Heatmap ---\n');

fig1 = figure('Name', 'IBS Connectivity Heatmap', ...
    'Color', 'w', 'Position', [50, 50, 1400, 400]);

% Load matrices for each class
matrices = cell(1, 3);
for i = 1:3
    filepath = fullfile(mean_by_class_dir, ...
        sprintf('%s_%s_%s.csv', classes{i}, key_band, key_feature));
    matrices{i} = load_matrix(filepath);
    if isempty(matrices{i})
        error('Could not load matrix for %s', classes{i});
    end
end

% Find global color limits for fair comparison
all_vals = [matrices{1}(:); matrices{2}(:); matrices{3}(:)];
clim_min = prctile(all_vals, 5);
clim_max = prctile(all_vals, 95);

% Plot each class
for i = 1:3
    subplot(1, 3, i);
    imagesc(matrices{i});
    colormap(gca, parula);
    clim([clim_min, clim_max]);
    axis square;

    % Labels
    title(classes{i}, 'FontSize', 14, 'FontWeight', 'bold', 'Color', colors_class(i,:));
    xlabel('Brain 2 Channels', 'FontSize', 11);
    ylabel('Brain 1 Channels', 'FontSize', 11);

    % Tick labels (show every 4th channel)
    tick_idx = 1:4:num_channels;
    set(gca, 'XTick', tick_idx, 'XTickLabel', channel_names(tick_idx), ...
        'YTick', tick_idx, 'YTickLabel', channel_names(tick_idx));
    xtickangle(45);

    colorbar;
end

sgtitle(sprintf('Inter-Brain %s Connectivity (%s band)', key_feature, key_band), ...
    'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig1, fullfile(output_dir, 'ibs_heatmap_3classes.png'));
fprintf('Saved: ibs_heatmap_3classes.png\n');


%% ========================================================================
% 2. Difference Matrix Heatmap (Cooperation - Competition)
% =========================================================================

fprintf('\n--- 2. Generating Difference Matrix Heatmap ---\n');

fig2 = figure('Name', 'IBS Difference Heatmap', ...
    'Color', 'w', 'Position', [100, 100, 900, 700]);

% Load difference matrix
diff_filepath = fullfile(diff_dir, sprintf('diff_%s_%s.csv', key_band, key_feature));
diff_matrix = load_matrix(diff_filepath);

if ~isempty(diff_matrix)
    % Create diverging colormap (blue-white-red)
    n = 256;
    half = n / 2;
    blue_to_white = [linspace(0.2, 1, half)', linspace(0.4, 1, half)', linspace(0.8, 1, half)'];
    white_to_red = [linspace(1, 0.9, half)', linspace(1, 0.3, half)', linspace(1, 0.3, half)'];
    diverging_cmap = [blue_to_white; white_to_red];

    % Symmetric color limits
    max_abs = max(abs(diff_matrix(:)));

    imagesc(diff_matrix);
    colormap(gca, diverging_cmap);
    clim([-max_abs, max_abs]);
    axis square;
    colorbar;

    % Labels
    title(sprintf('Cooperation - Competition (%s %s)', key_band, key_feature), ...
        'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Brain 2 Channels', 'FontSize', 12);
    ylabel('Brain 1 Channels', 'FontSize', 12);

    % Full tick labels
    set(gca, 'XTick', 1:num_channels, 'XTickLabel', channel_names, ...
        'YTick', 1:num_channels, 'YTickLabel', channel_names);
    xtickangle(90);
    set(gca, 'FontSize', 8);

    % Add text annotation
    text(0.02, 0.98, sprintf('Red: Coop > Comp\nBlue: Comp > Coop'), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', [0.7 0.7 0.7]);

    % Save
    saveas(fig2, fullfile(output_dir, 'ibs_diff_heatmap.png'));
    fprintf('Saved: ibs_diff_heatmap.png\n');
else
    warning('Difference matrix not found');
end


%% ========================================================================
% 3. Circular Connectivity Graph
% =========================================================================

fprintf('\n--- 3. Generating Circular Connectivity Graph ---\n');

fig3 = figure('Name', 'IBS Circular Graph', ...
    'Color', 'w', 'Position', [100, 100, 1500, 500]);

% Parameters for circular layout
theta_angles = linspace(0, 2*pi, num_channels + 1);
theta_angles = theta_angles(1:end-1);  % Remove last (duplicate of first)
radius = 1;

% Node positions
node_x = radius * cos(theta_angles);
node_y = radius * sin(theta_angles);

% Threshold for showing connections (top percentile)
threshold_pct = 95;

for class_idx = 1:3
    subplot(1, 3, class_idx);
    hold on;

    mat = matrices{class_idx};
    threshold_val = prctile(mat(:), threshold_pct);

    % Draw edges (connections above threshold)
    [row, col] = find(mat > threshold_val);
    for k = 1:length(row)
        i = row(k);
        j = col(k);
        if i ~= j  % Skip self-connections
            edge_weight = (mat(i,j) - threshold_val) / (max(mat(:)) - threshold_val);
            line_width = 0.5 + 2 * edge_weight;
            alpha_val = 0.3 + 0.5 * edge_weight;

            plot([node_x(i), node_x(j)], [node_y(i), node_y(j)], ...
                'Color', [colors_class(class_idx, :), alpha_val], ...
                'LineWidth', line_width);
        end
    end

    % Draw nodes
    scatter(node_x, node_y, 100, colors_class(class_idx, :), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1);

    % Add channel labels
    label_radius = 1.15;
    for i = 1:num_channels
        angle = theta_angles(i);
        text(label_radius * cos(angle), label_radius * sin(angle), ...
            channel_names{i}, 'HorizontalAlignment', 'center', ...
            'FontSize', 7, 'Rotation', rad2deg(angle) - 90);
    end

    % Styling
    axis equal;
    axis off;
    xlim([-1.5, 1.5]);
    ylim([-1.5, 1.5]);
    title(classes{class_idx}, 'FontSize', 14, 'FontWeight', 'bold', ...
        'Color', colors_class(class_idx, :));

    hold off;
end

sgtitle(sprintf('Inter-Brain Connectivity Graph (%s %s, top %d%%)', ...
    key_band, key_feature, 100-threshold_pct), ...
    'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig3, fullfile(output_dir, 'ibs_circular_graph.png'));
fprintf('Saved: ibs_circular_graph.png\n');


%% ========================================================================
% 4. ROI-based Statistics
% =========================================================================

fprintf('\n--- 4. Generating ROI-based Statistics ---\n');

fig4 = figure('Name', 'IBS ROI Analysis', ...
    'Color', 'w', 'Position', [100, 100, 1200, 800]);

% Get channel indices for each ROI
num_rois = length(roi_names);
roi_indices = cell(1, num_rois);
for r = 1:num_rois
    roi_idx = [];
    for ch = 1:length(roi_channels{r})
        idx = find(strcmpi(channel_names, roi_channels{r}{ch}));
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

% --- 4a: ROI Connectivity Matrices ---
for class_idx = 1:3
    subplot(2, 3, class_idx);
    imagesc(roi_connectivity(:, :, class_idx));
    colormap(gca, parula);
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

% --- 4b: ROI Bar Comparison ---
subplot(2, 3, 4:6);

% Extract diagonal (same-region connectivity) and off-diagonal means
same_region = zeros(3, num_rois);
cross_region = zeros(3, 1);
for class_idx = 1:3
    mat = roi_connectivity(:, :, class_idx);
    same_region(class_idx, :) = diag(mat);
    off_diag = mat(~eye(size(mat)));
    cross_region(class_idx) = mean(off_diag);
end

% Bar plot for same-region connectivity
bar_data = same_region';  % (num_rois x 3)
b = bar(bar_data);
for i = 1:3
    b(i).FaceColor = colors_class(i, :);
end

set(gca, 'XTickLabel', roi_names);
xlabel('Brain Region', 'FontSize', 12);
ylabel(sprintf('Mean %s', key_feature), 'FontSize', 12);
title('Same-Region Inter-Brain Connectivity by Class', 'FontSize', 14, 'FontWeight', 'bold');
legend(classes, 'Location', 'northeast');
grid on;

sgtitle(sprintf('ROI-based Inter-Brain Synchrony Analysis (%s %s)', key_band, key_feature), ...
    'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig4, fullfile(output_dir, 'ibs_roi_analysis.png'));
fprintf('Saved: ibs_roi_analysis.png\n');

% Save ROI statistics to CSV
roi_stats_table = array2table(same_region, 'VariableNames', roi_names, 'RowNames', classes);
writetable(roi_stats_table, fullfile(output_dir, 'ibs_roi_stats.csv'), 'WriteRowNames', true);
fprintf('Saved: ibs_roi_stats.csv\n');


%% ========================================================================
% 5. Multi-band Comparison
% =========================================================================

fprintf('\n--- 5. Generating Multi-band Comparison ---\n');

fig5 = figure('Name', 'IBS Multi-band Comparison', ...
    'Color', 'w', 'Position', [50, 50, 1400, 900]);

num_bands = length(bands);

% Compute mean connectivity for each band/class combination
band_means = zeros(num_bands, 3);
band_stds = zeros(num_bands, 3);

for b_idx = 1:num_bands
    for class_idx = 1:3
        filepath = fullfile(mean_by_class_dir, ...
            sprintf('%s_%s_%s.csv', classes{class_idx}, bands{b_idx}, key_feature));
        mat = load_matrix(filepath);
        if ~isempty(mat)
            band_means(b_idx, class_idx) = mean(mat(:));
            band_stds(b_idx, class_idx) = std(mat(:));
        end
    end
end

% --- 5a: Bar plot of mean connectivity by band ---
subplot(2, 2, 1);
b = bar(band_means);
for i = 1:3
    b(i).FaceColor = colors_class(i, :);
end
set(gca, 'XTickLabel', bands);
xlabel('Frequency Band', 'FontSize', 12);
ylabel(sprintf('Mean %s', key_feature), 'FontSize', 12);
title('Mean Inter-Brain Connectivity by Frequency Band', 'FontSize', 14, 'FontWeight', 'bold');
legend(classes, 'Location', 'northeast');
grid on;
xtickangle(45);

% --- 5b: Line plot showing band profiles ---
subplot(2, 2, 2);
hold on;
for class_idx = 1:3
    plot(1:num_bands, band_means(:, class_idx), '-o', ...
        'Color', colors_class(class_idx, :), ...
        'LineWidth', 2, 'MarkerSize', 8, ...
        'MarkerFaceColor', colors_class(class_idx, :));
end
hold off;
set(gca, 'XTick', 1:num_bands, 'XTickLabel', bands);
xlabel('Frequency Band', 'FontSize', 12);
ylabel(sprintf('Mean %s', key_feature), 'FontSize', 12);
title('Frequency Band Profile by Class', 'FontSize', 14, 'FontWeight', 'bold');
legend(classes, 'Location', 'best');
grid on;
xtickangle(45);

% --- 5c: Difference (Cooperation - Competition) by band ---
subplot(2, 2, 3);
diff_by_band = band_means(:, 3) - band_means(:, 2);  % Coop - Comp
bar_colors = zeros(num_bands, 3);
for i = 1:num_bands
    if diff_by_band(i) > 0
        bar_colors(i, :) = colors_class(3, :);  % Green for Coop > Comp
    else
        bar_colors(i, :) = colors_class(2, :);  % Orange for Comp > Coop
    end
end
bh = bar(diff_by_band);
bh.FaceColor = 'flat';
bh.CData = bar_colors;

hold on;
yline(0, '--k', 'LineWidth', 1);
hold off;

set(gca, 'XTickLabel', bands);
xlabel('Frequency Band', 'FontSize', 12);
ylabel(sprintf('\\Delta %s (Coop - Comp)', key_feature), 'FontSize', 12);
title('Cooperation vs Competition Difference by Band', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xtickangle(45);

% --- 5d: Heatmap of all bands x classes ---
subplot(2, 2, 4);
imagesc(band_means);
colormap(gca, parula);
colorbar;

set(gca, 'XTick', 1:3, 'XTickLabel', classes, ...
    'YTick', 1:num_bands, 'YTickLabel', bands);
xlabel('Class', 'FontSize', 12);
ylabel('Frequency Band', 'FontSize', 12);
title(sprintf('Mean %s Heatmap (Band x Class)', key_feature), 'FontSize', 14, 'FontWeight', 'bold');

% Add text values
for b_idx = 1:num_bands
    for class_idx = 1:3
        text(class_idx, b_idx, sprintf('%.3f', band_means(b_idx, class_idx)), ...
            'HorizontalAlignment', 'center', 'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
    end
end

sgtitle(sprintf('Multi-band Inter-Brain Synchrony Comparison (%s)', key_feature), ...
    'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig5, fullfile(output_dir, 'ibs_multiband_comparison.png'));
fprintf('Saved: ibs_multiband_comparison.png\n');

% Save band statistics to CSV
band_stats_table = array2table(band_means, 'VariableNames', classes, 'RowNames', bands);
writetable(band_stats_table, fullfile(output_dir, 'ibs_band_stats.csv'), 'WriteRowNames', true);
fprintf('Saved: ibs_band_stats.csv\n');


%% ========================================================================
% 6. Summary Figure (Publication-ready)
% =========================================================================

fprintf('\n--- 6. Generating Summary Figure ---\n');

fig6 = figure('Name', 'IBS Summary', ...
    'Color', 'w', 'Position', [50, 50, 1600, 1000]);

% --- Panel A: 3-class heatmaps (small) ---
for i = 1:3
    subplot(3, 4, i);
    imagesc(matrices{i});
    colormap(gca, parula);
    clim([clim_min, clim_max]);
    axis square;
    title(classes{i}, 'FontSize', 11, 'FontWeight', 'bold', 'Color', colors_class(i,:));
    set(gca, 'XTick', [], 'YTick', []);
    if i == 3
        cb = colorbar;
        cb.Label.String = key_feature;
    end
end

% --- Panel B: Difference matrix ---
subplot(3, 4, 4);
if ~isempty(diff_matrix)
    imagesc(diff_matrix);
    colormap(gca, diverging_cmap);
    clim([-max_abs, max_abs]);
    axis square;
    title('Coop - Comp', 'FontSize', 11, 'FontWeight', 'bold');
    set(gca, 'XTick', [], 'YTick', []);
    colorbar;
end

% --- Panel C: ROI analysis ---
subplot(3, 4, 5:6);
b = bar(same_region');
for i = 1:3
    b(i).FaceColor = colors_class(i, :);
end
set(gca, 'XTickLabel', roi_names);
ylabel(sprintf('Mean %s', key_feature));
title('ROI-based Connectivity', 'FontSize', 12, 'FontWeight', 'bold');
legend(classes, 'Location', 'eastoutside', 'FontSize', 8);
grid on;
xtickangle(30);

% --- Panel D: Multi-band profile ---
subplot(3, 4, 7:8);
hold on;
for class_idx = 1:3
    plot(1:num_bands, band_means(:, class_idx), '-o', ...
        'Color', colors_class(class_idx, :), ...
        'LineWidth', 2, 'MarkerSize', 6, ...
        'MarkerFaceColor', colors_class(class_idx, :));
end
hold off;
set(gca, 'XTick', 1:num_bands, 'XTickLabel', bands);
ylabel(sprintf('Mean %s', key_feature));
title('Frequency Band Profile', 'FontSize', 12, 'FontWeight', 'bold');
legend(classes, 'Location', 'eastoutside', 'FontSize', 8);
grid on;
xtickangle(30);

% --- Panel E: Circular graphs (smaller) ---
for class_idx = 1:3
    subplot(3, 4, 8 + class_idx);
    hold on;

    mat = matrices{class_idx};
    threshold_val = prctile(mat(:), 97);

    [row, col] = find(mat > threshold_val);
    for k = 1:length(row)
        i_idx = row(k);
        j_idx = col(k);
        if i_idx ~= j_idx
            plot([node_x(i_idx), node_x(j_idx)], [node_y(i_idx), node_y(j_idx)], ...
                'Color', [colors_class(class_idx, :), 0.4], 'LineWidth', 0.8);
        end
    end

    scatter(node_x, node_y, 40, colors_class(class_idx, :), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);

    axis equal; axis off;
    xlim([-1.3, 1.3]); ylim([-1.3, 1.3]);
    title(classes{class_idx}, 'FontSize', 10, 'Color', colors_class(class_idx, :));
    hold off;
end

% --- Panel F: Statistics text ---
subplot(3, 4, 12);
axis off;

% Find most different ROI
[~, max_roi_idx] = max(abs(same_region(3,:) - same_region(2,:)));
max_roi_diff = same_region(3, max_roi_idx) - same_region(2, max_roi_idx);

% Find most different band
[~, max_band_idx] = max(abs(diff_by_band));
max_band_diff = diff_by_band(max_band_idx);

stats_text = {
    '=== Key Findings ===', ...
    '', ...
    sprintf('Band: %s', key_band), ...
    sprintf('Feature: %s', key_feature), ...
    '', ...
    '--- Global Mean ---', ...
    sprintf('  Single: %.4f', mean(matrices{1}(:))), ...
    sprintf('  Comp:   %.4f', mean(matrices{2}(:))), ...
    sprintf('  Coop:   %.4f', mean(matrices{3}(:))), ...
    '', ...
    '--- Largest ROI Diff ---', ...
    sprintf('  %s: %.4f', roi_names{max_roi_idx}, max_roi_diff), ...
    '', ...
    '--- Largest Band Diff ---', ...
    sprintf('  %s: %.4f', bands{max_band_idx}, max_band_diff)
};

text(0.1, 0.95, stats_text, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontSize', 9, 'FontName', 'FixedWidth');

sgtitle(sprintf('Inter-Brain Synchrony Analysis Summary (%s %s)', key_band, key_feature), ...
    'FontSize', 16, 'FontWeight', 'bold');

% Save
saveas(fig6, fullfile(output_dir, 'ibs_summary.png'));
fprintf('Saved: ibs_summary.png\n');


%% ========================================================================
% Done
% =========================================================================

fprintf('\n=== IBS Connectivity Analysis Complete ===\n');
fprintf('All figures saved to: %s\n', output_dir);
fprintf('\nGenerated files:\n');
fprintf('  1. ibs_heatmap_3classes.png     - Connectivity matrices\n');
fprintf('  2. ibs_diff_heatmap.png         - Difference matrix\n');
fprintf('  3. ibs_circular_graph.png       - Circular connectivity\n');
fprintf('  4. ibs_roi_analysis.png         - ROI-based analysis\n');
fprintf('  5. ibs_multiband_comparison.png - Multi-band comparison\n');
fprintf('  6. ibs_summary.png              - Publication summary\n');
fprintf('  7. ibs_roi_stats.csv            - ROI statistics\n');
fprintf('  8. ibs_band_stats.csv           - Band statistics\n');


