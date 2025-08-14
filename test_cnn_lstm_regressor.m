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

function test_model(net, test_rat, mu_y, sigma_y, ...
                    all_norms, all_speeds, all_entropies, ...
                    running_times, running_speeds, ...
                    activity_timestamps, lag, half_window_size, sequenceLength)

    % Extract sequences for the test rat
    dysco_time_test = activity_timestamps{test_rat}(half_window_size+1:end-half_window_size);
    [X_test, y_test] = extract_sequence_features( ...
        all_norms{test_rat}, all_speeds{test_rat}, all_entropies{test_rat}, ...
        running_times{test_rat}, running_speeds{test_rat}, ...
        dysco_time_test, lag, sequenceLength);

    % Normalize target using training mean and std
    y_test_z = (y_test - mu_y) / sigma_y;

    % Predict with the trained network
    y_pred_z = predict(net, X_test);

    % Denormalize predictions
    y_pred = y_pred_z * sigma_y + mu_y;

    % Compute performance metrics
    rmse = sqrt(mean((y_pred - y_test).^2));
    r = corr(y_pred, y_test, 'rows', 'complete');
    r2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);

    % Display results
    fprintf('Test Rat %d — RMSE: %.3f | r: %.3f | R²: %.3f\n', test_rat, rmse, r, r2);

    % Plot true vs predicted
    figure;
    plot(y_test, 'k', 'DisplayName', 'True');
    hold on;
    plot(y_pred, 'r', 'DisplayName', 'Predicted');
    legend();
    title(sprintf('Test Rat %d | RMSE %.2f | r %.2f | R² %.2f', test_rat, rmse, r, r2));
    xlabel('Time');
    ylabel('Running Speed');
    grid on;
end

test_rat_id = 3; % example rat to test
test_model(net, test_rat_id, mu_y, sigma_y, ...
           all_norms, all_speeds, all_entropies, ...
           running_times, running_speeds, ...
           activity_timestamps, lag, half_window_size, sequenceLength);
