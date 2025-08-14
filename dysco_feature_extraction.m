% if you want to learn how to use dysco this is the right place! This
% scripts teaches you how to run the core functions to build your dysco
% analysis pipeline

%clear,close
addpath("../core_functions");

% Define DySCo parameters
half_window_size = 10;
n_eigen = 10;
lag = 5;

% Preallocate output cell arrays
all_norms = cell(1, length(activity_data));
all_entropies = cell(1, length(activity_data));
all_FCDs = cell(1, length(activity_data));
all_speeds = cell(1, length(activity_data));
eigenvectors = cell(1, length(activity_data));
eigenvalues = cell(1, length(activity_data));
avg_mini_truncated_matrix = cell(1, length(activity_data));
for i = 1:length(activity_data)
    data = activity_data{i};
    if isempty(data)
        fprintf('Skipping empty data at index %d\\n', i);
        continue;
    end

    try
        % EVD of sliding correlation matrix
        [eigenvectors{i}, eigenvalues{i}, avg_mini_truncated_matrix{i}] = compute_eigenvectors_sliding_corr_modified(data, half_window_size, n_eigen);

        % DySCo Norm
        norm = dysco_norm(eigenvalues{i}, 2);
        all_norms{i} = norm;

        % Von Neumann Entropy
        entropy = dysco_entropy(eigenvalues{i});
        all_entropies{i} = entropy;

        % Functional Connectivity Dynamics (FCD)
        T = size(eigenvalues{i}, 2);
        FCD = zeros(T, T);
        for m = 1:T
            for n = m+1:T
                FCD(m,n) = dysco_distance(eigenvectors{i}(:,:,m), eigenvectors{i}(:,:,n), 2);
                FCD(n,m) = FCD(m,n);
            end
        end
        all_FCDs{i} = FCD;

        % Reconfiguration Speed
        speed = zeros(1, T - lag);
        for t = 1:T - lag
            speed(t) = FCD(t, t + lag);
        end
        all_speeds{i} = speed;

        fprintf('Processed DySCo for session %d\\n', i);
    catch ME
        fprintf('Error in session %d: %s\\n', i, ME.message);
    end
end

