function results = mixed_model_classification_dysco(all_norms, all_speeds, all_entropies, ...
    activity_timestamps, running_times, running_speeds, half_window_size)

    % Exclude bad rats
    valid_rats = setdiff(1:10, [1 4]);
    lag = 1;  
    smooth_win = 10;  % smoothing window for running speed

    X_all = cell(1, 10);
    y_class_all = cell(1, 10);

    % === Feature extraction per rat ===
    for rat_no = valid_rats
        dysco_time = activity_timestamps{rat_no}(half_window_size+1:end-half_window_size);

        % Interpolate & smooth running speed
        run_speed_interp = interp1(running_times{rat_no}, running_speeds{rat_no}, ...
                                   dysco_time(1:end-lag), 'linear', 'extrap');
        run_speed_smooth = movmean(run_speed_interp, smooth_win);
        run_speed_smooth(run_speed_smooth < 0) = 0;

        % Binary label: running vs stationary
        binary_label = double(run_speed_smooth > 2);

        % Truncate DySCo features to match speed length
        min_len = min([length(all_norms{rat_no}), length(all_speeds{rat_no}), ...
                       length(all_entropies{rat_no}), length(run_speed_smooth)]);

        norm_feat = all_norms{rat_no}(1:min_len);
        speed_feat = all_speeds{rat_no}(1:min_len);
        entropy_feat = all_entropies{rat_no}(1:min_len);

        % Rolling stats
        win = 3;
        norm_mean = movmean(norm_feat, [win 0]);
        speed_mean = movmean(speed_feat, [win 0]);
        entropy_mean = movmean(entropy_feat, [win 0]);

        norm_std = movstd(norm_feat, [win 0]);
        speed_std = movstd(speed_feat, [win 0]);
        entropy_std = movstd(entropy_feat, [win 0]);

        % Combine features
        features = [norm_feat(:), speed_feat(:), entropy_feat(:), ...
                    norm_mean(:), speed_mean(:), entropy_mean(:), ...
                    norm_std(:), speed_std(:), entropy_std(:)];

        X_all{rat_no} = features;
        y_class_all{rat_no} = binary_label(1:min_len);
    end

    % === Models ===
    models = {
        @fitcsvm, ...
        @fitcensemble, ...
        @(X, y) fitclinear(X, y, 'Learner', 'logistic', 'Solver', 'lbfgs')
    };
    model_names = {'SVM', 'RF', 'Logistic'};

    result_struct = struct();
    y_preds_all = cell(1, length(models));
    y_truth_all = cell(1, length(models));

    % === Cross-validation (Leave-One-Rat-Out) ===
    for m = 1:length(models)
        accs = []; aucs = [];
        y_all_preds = [];
        y_all_truth = [];

        for test_rat = valid_rats
            if isempty(X_all{test_rat}), continue; end

            X_test = X_all{test_rat};
            y_test = y_class_all{test_rat};

            X_train = []; y_train = [];
            for train_rat = valid_rats
                if train_rat == test_rat || isempty(X_all{train_rat}), continue; end
                min_len = min(size(X_test, 1), size(X_all{train_rat}, 1));
                X_train = [X_train; X_all{train_rat}(1:min_len, :)];
                y_train = [y_train; y_class_all{train_rat}(1:min_len)];
            end

            min_len = min(size(X_train, 1), size(X_test, 1));
            X_train = X_train(1:min_len, :);
            y_train = y_train(1:min_len);
            X_test = X_test(1:min_len, :);
            y_test = y_test(1:min_len);


            % Standardize
            mu = mean(X_train);
            sigma = std(X_train);
            X_train = (X_train - mu) ./ (sigma + eps);
            X_test = (X_test - mu) ./ (sigma + eps);

            % Train & predict
            clf = models{m}(X_train, y_train);
            y_pred = predict(clf, X_test);
            acc = mean(y_pred == y_test);
            [~, ~, ~, auc] = perfcurve(y_test, double(y_pred), 1);

            accs(end+1) = acc;
            aucs(end+1) = auc;
            y_all_preds = [y_all_preds; y_pred(:)];
            y_all_truth = [y_all_truth; y_test(:)];
        end

        result_struct.(model_names{m}) = struct( ...
            'Accuracy', mean(accs), ...
            'AUC', mean(aucs));
        y_preds_all{m} = y_all_preds;
        y_truth_all{m} = y_all_truth;
    end

    result_struct.y_preds = y_preds_all;
    result_struct.y_truth = y_truth_all;
    results = result_struct;

    % === Select best model by AUC ===
    auc_vals = [results.SVM.AUC, results.RF.AUC, results.Logistic.AUC];
    [~, best_idx] = max(auc_vals);
    best_model = model_names{best_idx};

    y_true = results.y_truth{best_idx};
    y_pred = results.y_preds{best_idx};

    % === Metrics ===
    acc = mean(y_pred == y_true);
    f1 = f1_score(y_true, y_pred);
    prec = precision_score(y_true, y_pred);
    rec = recall_score(y_true, y_pred);
    R2 = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);

    fprintf('\n--- Best Model: %s ---\n', best_model);
    fprintf('Accuracy: %.4f\n', acc);
    fprintf('AUC: %.4f\n', auc_vals(best_idx));
    fprintf('F1 Score: %.4f\n', f1);
    fprintf('Precision: %.4f\n', prec);
    fprintf('Recall: %.4f\n', rec);
    fprintf('R²: %.4f\n', R2);

    % === Confusion Matrix ===
    figure;
    confusionchart(y_true, y_pred);
    title(['Confusion Matrix — ', best_model]);

    % === Prediction vs True Plot ===
    figure;
    plot(y_true, 'k-', 'DisplayName', 'True');
    hold on;
    plot(y_pred, 'r--', 'DisplayName', 'Predicted');
    xlabel('Time Index');
    ylabel('Class (0 = stationary, 1 = running)');
    title(['Predicted vs True — ', best_model]);
    legend;
    grid on;
end

% % === Metric helper functions ===
function f1 = f1_score(y_true, y_pred)
    tp = sum((y_true == 1) & (y_pred == 1));
    fp = sum((y_true == 0) & (y_pred == 1));
    fn = sum((y_true == 1) & (y_pred == 0));
    if tp + fp == 0 || tp + fn == 0
        f1 = NaN;
    else
        prec = tp / (tp + fp);
        rec = tp / (tp + fn);
        f1 = 2 * (prec * rec) / (prec + rec);
    end
end

function p = precision_score(y_true, y_pred)
    tp = sum((y_true == 1) & (y_pred == 1));
    fp = sum((y_true == 0) & (y_pred == 1));
    if tp + fp == 0
        p = NaN;
    else
        p = tp / (tp + fp);
    end
end

function r = recall_score(y_true, y_pred)
    tp = sum((y_true == 1) & (y_pred == 1));
    fn = sum((y_true == 1) & (y_pred == 0));
    if tp + fn == 0
        r = NaN;
    else
        r = tp / (tp + fn);
    end
end



 % Train & Evaluate
    results = mixed_model_classification_dysco(all_norms, all_speeds, all_entropies, ...
        activity_timestamps, running_times, running_speeds, half_window_size);

    % Model names
    model_names = {'SVM', 'RF', 'Logistic'};
    auc_vals = [results.SVM.AUC, results.RF.AUC, results.Logistic.AUC];
    [~, best_idx] = max(auc_vals);
    best_model = model_names{best_idx};

    % Best model predictions
    y_true = results.y_truth{best_idx};
    y_pred = results.y_preds{best_idx};

    % Metrics
    acc = mean(y_pred == y_true);
    f1 = f1_score(y_true, y_pred);
    prec = precision_score(y_true, y_pred);
    rec = recall_score(y_true, y_pred);
    R2 = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);

    % Print metrics
    fprintf('\n--- Best Model: %s ---\n', best_model);
    fprintf('Accuracy: %.4f\n', acc);
    fprintf('AUC: %.4f\n', auc_vals(best_idx));
    fprintf('F1 Score: %.4f\n', f1);
    fprintf('Precision: %.4f\n', prec);
    fprintf('Recall: %.4f\n', rec);
    fprintf('R²: %.4f\n', R2);

    % Confusion Matrix
    figure;
    confusionchart(y_true, y_pred);
    title(['Confusion Matrix — ', best_model]);

    % Predicted vs True Plot
    figure;
    plot(y_true, 'k-', 'DisplayName', 'True');
    hold on;
    plot(y_pred, 'r--', 'DisplayName', 'Predicted');
    xlabel('Time Index');
    ylabel('Class (0 = stationary, 1 = running)');
    title(['Predicted vs True — ', best_model]);
    legend;
    grid on;