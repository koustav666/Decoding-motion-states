function [eigenvectors, eigenvalues, avg_mini_truncated_matrix] = compute_eigenvectors_sliding_corr_modified(timeseries, half_window_size, n_eigen)

    if n_eigen > 2*half_window_size
        error('Number of requested eigenvectors is too large');
    end

    N = size(timeseries,2);
    T = size(timeseries,1);
    n_windows = T - 2*half_window_size;

    eigenvectors = zeros(N, n_eigen, n_windows);
    eigenvalues = zeros(n_eigen, n_windows);
    avg_mini_truncated_matrix = zeros(N, N, n_windows);  % NEW: For CNN input

    for t = 1:n_windows
        truncated_timeseries = timeseries(t:t + 2*half_window_size, :);
        zscored_truncated = zscore(truncated_timeseries);
        normalizing_factor = size(truncated_timeseries,1) - 1;
        zscored_truncated = (1 / sqrt(normalizing_factor)) * zscored_truncated;

        minimatrix = zscored_truncated * zscored_truncated';
        [v, lambda] = eigs(minimatrix, n_eigen);

        eigenvalues(:, t) = diag(lambda);
        eig_vecs = zscored_truncated' * v;

        for i = 1:n_eigen
            eig_vecs(:, i) = eig_vecs(:, i) / norm(eig_vecs(:, i));
            eig_vecs(:, i) = eig_vecs(:, i) * sqrt(eigenvalues(i, t));
        end

        eigenvectors(:, :, t) = eig_vecs;

        % --- New: Store average connectivity matrix for CNN
        avg_mini_truncated_matrix(:, :, t) = zscored_truncated' * zscored_truncated;
    end
end
