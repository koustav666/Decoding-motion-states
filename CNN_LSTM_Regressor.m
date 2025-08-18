function [X_seq, y_out] = extract_sequence_features(norm, speed, entropy, ...
    run_times, run_speeds, dysco_time, lag, seq_len)

    % Truncate
    norm = norm(1:end-lag)';
    speed = speed(:);
    entropy = entropy(1:end-lag)';
    t = dysco_time(1:end-lag);

    % Interpolated speed
    y = interp1(run_times, run_speeds, t, 'linear', 'extrap');
    y(y <= 0) = 0.01;

    % Derived features
    d_norm = [0; diff(norm)];
    d_speed = [0; diff(speed)];
    d_entropy = [0; diff(entropy)];
    win = 5;
    roll_mean_norm = movmean(norm, win);
    roll_std_norm = movstd(norm, win);
    roll_mean_speed = movmean(speed, win);
    roll_std_speed = movstd(speed, win);
    interaction = norm .* speed;

    % Full feature matrix
    X_full = [norm, speed, entropy, ...
              d_norm, d_speed, d_entropy, ...
              roll_mean_norm, roll_std_norm, ...
              roll_mean_speed, roll_std_speed, ...
              interaction];

    % Normalize features
    X_full = (X_full - mean(X_full)) ./ std(X_full);

    % Extract sequences
    n = size(X_full, 1);
    X_seq = {};
    y_out = [];
    for i = 1:(n - seq_len + 1)
        X_seq{end+1,1} = X_full(i:i+seq_len-1, :);
        y_out(end+1,1) = y(i+seq_len-1);  % predict final value
    end
end
valid_rats = setdiff(1:length(all_norms), [1, 4]);
lag = 5;
half_window_size = 10;
sequenceLength = 100;  % Use short history
featureDim = 100;

all_r = zeros(length(valid_rats), 1);
all_rmse = zeros(length(valid_rats), 1);
all_r2 = zeros(length(valid_rats), 1);

% --- Define CNN+LSTM model ---
layers = [
    sequenceInputLayer(featureDim, "Name", "input")

    convolution1dLayer(3, 32, "Padding", "same", "Name", "conv1")
    batchNormalizationLayer("Name", "bn1")
    reluLayer("Name", "relu1")

    lstmLayer(32, "OutputMode", "last", "Name", "lstm")
    fullyConnectedLayer(32, "Name", "fc1")
    reluLayer("Name", "relu2")
    dropoutLayer(0.3, "Name", "drop")

    fullyConnectedLayer(1, "Name", "output")
    regressionLayer("Name", "regression")
];

options = trainingOptions("adam", ...
    "MaxEpochs", 15, ...
    "MiniBatchSize", 64, ...
    "Shuffle", "every-epoch", ...
    "Verbose", false, ...
    "Plots", "training-progress");

% Initialize dummy model
dummyX = repmat({rand(sequenceLength, featureDim)}, 5, 1);
dummyY = rand(5, 1);
net = trainNetwork(dummyX, dummyY, layers, options);

% --- Leave-One-Rat-Out Loop ---
for i = 1:length(valid_rats)
    val_rat = valid_rats(i);
    train_rats = setdiff(valid_rats, val_rat);

    X_train_full = {}; Y_train_full = [];
    for rat = train_rats
        dysco_time = activity_timestamps{rat}(half_window_size+1:end-half_window_size);
        [X_seq, y] = extract_sequence_features( ...
            all_norms{rat}, all_speeds{rat}, all_entropies{rat}, ...
            running_times{rat}, running_speeds{rat}, ...
            dysco_time, lag, sequenceLength);

        X_train_full = [X_train_full; X_seq];
        Y_train_full = [Y_train_full; y];
    end

    % --- Create validation split (80% train / 20% val) ---
    cv = cvpartition(length(Y_train_full), 'HoldOut', 0.2);
    X_train = X_train_full(training(cv));
    y_train = Y_train_full(training(cv));
    X_val = X_train_full(test(cv));
    y_val_split = Y_train_full(test(cv));

    % --- Normalize target ---
    mu_y = mean(y_train);
    sigma_y = std(y_train);
    y_train_z = (y_train - mu_y) / sigma_y;
    y_val_z = (y_val_split - mu_y) / sigma_y;

    % --- Train with validation data ---
    options.ValidationData = {X_val, y_val_z};
    net = trainNetwork(X_train, y_train_z, net.Layers, options);

    % --- Evaluate on held-out rat ---
    dysco_time_val = activity_timestamps{val_rat}(half_window_size+1:end-half_window_size);
    [X_test, y_test] = extract_sequence_features( ...
        all_norms{val_rat}, all_speeds{val_rat}, all_entropies{val_rat}, ...
        running_times{val_rat}, running_speeds{val_rat}, ...
        dysco_time_val, lag, sequenceLength);

    y_pred_z = predict(net, X_test);
    y_pred = y_pred_z * sigma_y + mu_y;

    % --- Metrics ---
    rmse = sqrt(mean((y_pred - y_test).^2));
    r = corr(y_pred, y_test, 'rows', 'complete');
    r2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);

    all_rmse(i) = rmse;
    all_r(i) = r;
    all_r2(i) = r2;

    fprintf('Rat %d — RMSE: %.3f | r: %.3f | R²: %.3f\n', val_rat, rmse, r, r2);

    % --- Optional plot ---
    figure;
    plot(y_test, 'k', 'DisplayName', 'True');
    hold on;
    plot(y_pred, 'r', 'DisplayName', 'Predicted');
    legend();
    title(sprintf('Rat %d | RMSE %.2f | r %.2f | R² %.2f', val_rat, rmse, r, r2));
    xlabel('Time'); ylabel('Running Speed'); grid on;
end

% --- Summary ---
fprintf('\n Final Leave-One-Rat-Out Summary:\n');
fprintf('Avg RMSE: %.4f ± %.4f\n', mean(all_rmse), std(all_rmse));
fprintf('Avg r:    %.4f ± %.4f\n', mean(all_r), std(all_r));
fprintf('Avg R²:   %.4f ± %.4f\n', mean(all_r2), std(all_r2));
