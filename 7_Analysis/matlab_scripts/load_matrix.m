%% ========================================================================
% Helper Function: Load Connectivity Matrix
% =========================================================================

function mat = load_matrix(filepath)
    if exist(filepath, 'file')
        mat = readmatrix(filepath);
    else
        mat = [];
        warning('File not found: %s', filepath);
    end
end