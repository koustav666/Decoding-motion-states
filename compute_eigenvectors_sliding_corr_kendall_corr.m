function [eigenvectors, eigenvalues, kendall_corr_matrix] = compute_eigenvectors_sliding_corr_modified(timeseries, half_window_size, n_eigen)
% This function computes eigenvectors, eigenvalues, and the underlying correlation
% matrix from a time-series using a sliding window and Kendall's Tau correlation.

    % Get the dimensions of the input time-series data
    [T, N] = size(timeseries);

    % --- Input Validation ---
    % Ensure the number of requested eigenvectors is not greater than the number of ROIs
    if n_eigen > N
        error('Number of requested eigenvectors cannot be larger than the number of time-series (N).');
    end

    % Calculate the total number of sliding windows
    n_windows = T - 2 * half_window_size;

    % --- Pre-allocation for Speed ---
    % Pre-allocate memory for the output variables
    eigenvectors = zeros(N, n_eigen, n_windows);
    eigenvalues = zeros(n_eigen, n_windows);
    kendall_corr_matrix = zeros(N, N, n_windows); % Store the Kendall matrix for each window

    % --- Sliding Window Loop ---
    % Iterate through each possible window in the time-series
    for t = 1:n_windows
        % Extract the segment of the time-series for the current window
        truncated_timeseries = timeseries(t:(t + 2 * half_window_size), :);

        % --- Step 1: Calculate Kendall's Tau Correlation Matrix ---
        % This is the key change. We compute the N x N correlation matrix
        % between all pairs of ROIs using Kendall's Tau.

        zscored_truncated = zscore(truncated_timeseries);
        normalizing_factor = size(truncated_timeseries,1) - 1;
        zscored_truncated = (1 / sqrt(normalizing_factor)) * zscored_truncated;



        % minimatrix = zscored_truncated * zscored_truncated';
        current_corr_matrix = corr(zscored_truncated, 'Type', 'Kendall', 'rows', 'pairwise');
        
        % Store the calculated correlation matrix
        %kendall_corr_matrix(:, :, t) = current_corr_matrix;

        % --- Step 2: Perform Eigendecomposition ---
        % Directly compute the top 'n_eigen' eigenvectors and eigenvalues
        % from the N x N Kendall correlation matrix.
        % We use 'eigs' for efficiency, as it only computes the requested number of eigenvalues.
        % 'largestabs' finds the eigenvalues with the largest magnitudes.
        [v, lambda] = eigs(current_corr_matrix, n_eigen, 'largestabs');

        % Store the results for the current window
        eigenvalues(:, t) = diag(lambda); % Store the diagonal of the eigenvalue matrix
        eigenvectors(:, :, t) = v;         % Store the eigenvectors
    end
end