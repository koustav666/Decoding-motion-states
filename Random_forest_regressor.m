function consolidated_model_report = crossrat_leaveoneout_rf(all_norms, all_speeds, all_entropies, running_times, running_speeds, activity_timestamps, lag, half_window_size)
    valid_rats = setdiff(1:length(all_norms), [1, 4]);  % skip rats 1 and 4
    n_valid = length(valid_rats);
    results = struct();

    % --- Leave-One-Rat-Out Evaluation ---
    for idx = 1:n_valid
        test_rat = valid_rats(idx);

        % --- Initialize training buffers
        X_train_all = [];
        y_train_all = [];

        % --- Loop through all rats except test_rat
        for train_rat = valid_rats
            if train_rat == test_rat
                continue;
            end

            dysco_time = activity_timestamps{train_rat}(half_window_size+1:end-half_window_size);
            [X, y] = extract_features_and_target(all_norms{train_rat}, ...
                                                 all_speeds{train_rat}, ...
                                                 all_entropies{train_rat}, ...
                                                 running_times{train_rat}, ...
                                                 running_speeds{train_rat}, ...
                                                 dysco_time, lag);
            if ~isempty(X) && ~isempty(y)
                X_train_all = [X_train_all; X];
                y_train_all = [y_train_all; y];
            end
        end

        % --- Train Random Forest model
        rf_model = TreeBagger(1000, X_train_all, y_train_all, ...
            'Method', 'regression', ...
            'OOBPrediction', 'On', ...
            'MinLeafSize', 10);

        % --- Test on held-out rat
        dysco_time_test = activity_timestamps{test_rat}(half_window_size+1:end-half_window_size);
        [X_test, y_test] = extract_features_and_target(all_norms{test_rat}, ...
                                                       all_speeds{test_rat}, ...
                                                       all_entropies{test_rat}, ...
                                                       running_times{test_rat}, ...
                                                       running_speeds{test_rat}, ...
                                                       dysco_time_test, lag);

        % --- Predict and evaluate
        y_pred = predict(rf_model, X_test);
        rmse = sqrt(mean((y_pred - y_test).^2));
        r = corr(y_pred, y_test);
        r2 = 1 - (sum((y_pred - y_test).^2) / sum((y_test - mean(y_test)).^2));

        % --- Store results
        results(test_rat).rmse = rmse;
        results(test_rat).r = r;
        results(test_rat).y_true = y_test;
        results(test_rat).y_pred = y_pred;
        results(test_rat).r2 = r2;

        % --- Plot prediction
        figure;
        plot(y_test, 'k', 'DisplayName', 'True Speed');
        hold on;
        plot(y_pred, 'r', 'DisplayName', 'Predicted');
        legend();
        xlabel('Sample');
        ylabel('Running Speed');
        title(sprintf('Test Rat %d | RMSE: %.3f | r: %.3f | R²: %.3f', test_rat, rmse, r, r2));
    end

    % --- Summary of cross-validation
    rmse_list = arrayfun(@(x) x.rmse, results(valid_rats));
    r_list = arrayfun(@(x) x.r, results(valid_rats));
    r2_list = arrayfun(@(x) x.r2, results(valid_rats));

    fprintf("\n===== Cross-Rat LOOCV Summary =====\n");
    for rat_idx = valid_rats
        res = results(rat_idx);
        fprintf("Rat %d | RMSE = %.4f | r = %.4f | R² = %.4f\n", ...
            rat_idx, res.rmse, res.r, res.r2);
    end
    fprintf("Average RMSE: %.4f ± %.4f\n", mean(rmse_list), std(rmse_list));
    fprintf("Average r:    %.4f ± %.4f\n", mean(r_list), std(r_list));
    fprintf("Average R²:   %.4f ± %.4f\n", mean(r2_list), std(r2_list));

    % --- Train final consolidated model on all valid rats
    X_all = [];
    y_all = [];
    for rat = valid_rats
        dysco_time = activity_timestamps{rat}(half_window_size+1:end-half_window_size);
        [X, y] = extract_features_and_target(all_norms{rat}, ...
                                             all_speeds{rat}, ...
                                             all_entropies{rat}, ...
                                             running_times{rat}, ...
                                             running_speeds{rat}, ...
                                             dysco_time, lag);
        if ~isempty(X) && ~isempty(y)
            X_all = [X_all; X];
            y_all = [y_all; y];
        end
    end

    consolidated_rf_model = TreeBagger(1000, X_all, y_all, ...
        'Method', 'regression', ...
        'OOBPrediction', 'On', ...
        'MinLeafSize', 10);

    % --- Output final report
    consolidated_model_report = struct( ...
        'results', results, ...
        'avg_rmse', mean(rmse_list), ...
        'avg_r', mean(r_list), ...
        'avg_r2', mean(r2_list), ...
        'rf_model', consolidated_rf_model ...
    );
end

function [X, y] = extract_features_and_target(norm, speed, entropy, running_times, running_speeds, dysco_time, lag)
    % Trim to match lag
    norm_trunc = norm(1:end-lag)';
    speed_trunc = speed(:);
    entropy_trunc = entropy(1:end-lag)';

    % Temporal gradients
    d_norm = [0; diff(norm_trunc)];
    d_speed = [0; diff(speed_trunc)];
    d_entropy = [0; diff(entropy_trunc)];

    % Rolling statistics
    win = 5;
    roll_mean_norm = movmean(norm_trunc, win);
    roll_std_norm  = movstd(norm_trunc, win);
    roll_mean_speed = movmean(speed_trunc, win);
    roll_std_speed  = movstd(speed_trunc, win);

    % Interaction feature
    interaction = norm_trunc .* speed_trunc;

    % Feature matrix
    X = [norm_trunc, speed_trunc, entropy_trunc, ...
         d_norm, d_speed, d_entropy, ...
         roll_mean_norm, roll_std_norm, ...
         roll_mean_speed, roll_std_speed, ...
         interaction];

    % Interpolated ground truth
    y = interp1(running_times, running_speeds, dysco_time(1:end-lag), 'linear', 'extrap');
    y(y <= 0) = 0.01;
    % Trim if mismatch
    min_len = min(size(X,1), length(y));
    X = X(1:min_len, :);
    y = y(1:min_len);

    % Optional: scale features + target (standardization)
    X = zscore(X);
    y = zscore(y);  % comment this line out if you don’t want scaled output
end

lag = 5;
half_window_size = 10;

best_model_report = crossrat_leaveoneout_rf(all_norms, all_speeds, all_entropies, ...
    running_times, running_speeds, activity_timestamps, lag, half_window_size);
