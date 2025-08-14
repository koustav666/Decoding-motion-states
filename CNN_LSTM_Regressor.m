function net = train_cnn_lstm_regression(all_norms, all_speeds, all_entropies, ...
                                         running_times, running_speeds, ...
                                         activity_timestamps, lag, half_window_size)

    valid_rats = setdiff(1:length(all_norms), [1, 4]);  % skip bad rats
    seq_len = 15;

    X_all = {}; Y_all = [];

    % === 1. Prepare sequences from all rats ===
    for rat = valid_rats
        if isempty(all_norms{rat}), continue; end
        dysco_time = activity_timestamps{rat}(half_window_size+1:end-half_window_size);
        [Xrat, Yrat] = prepare_dysco_sequences_for_regression( ...
            all_norms{rat}, all_speeds{rat}, all_entropies{rat}, ...
            running_times{rat}, running_speeds{rat}, dysco_time, lag, seq_len);
        if ~isempty(Xrat)
            X_all = [X_all; Xrat];
            Y_all = [Y_all; Yrat];
        end
    end

    % === 2. Normalize features across dataset ===
    allX = cat(1, X_all{:});
    mu = mean(allX, [1 2]);  % across all sequences and time steps
    sigma = std(allX, [], [1 2]) + eps;

    for i = 1:numel(X_all)
        X_all{i} = (X_all{i} - mu) ./ sigma;
    end

    % === 3. Split Train/Test (80/20) ===
    N = numel(Y_all);
    idx = randperm(N);
    Ntrain = round(0.8 * N);
    XTrain = X_all(idx(1:Ntrain));
    YTrain = Y_all(idx(1:Ntrain));
    XTest  = X_all(idx(Ntrain+1:end));
    YTest  = Y_all(idx(Ntrain+1:end));

    % === 4. Define Regression Network ===
    featureDim = size(XTrain{1}, 2);  % should be 3: norm, speed, entropy

    layers = [
        sequenceInputLayer(featureDim, "Name", "input")

        convolution1dLayer(3, 32, "Padding", "same", "Name", "conv1")
        batchNormalizationLayer("Name", "bn1")
        reluLayer("Name", "relu1")

        lstmLayer(32, "OutputMode", "last", "Name", "lstm")

        fullyConnectedLayer(1, "Name", "fc")
        regressionLayer("Name", "regressionOutput")
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    % === 5. Train ===
    net = trainNetwork(XTrain, YTrain, layers, options);

    % === 6. Predict and plot ===
    YPred = predict(net, XTest);
    rmse = sqrt(mean((YPred - YTest).^2));
    r = corr(YPred, YTest);

    fprintf('\n✅ CNN-LSTM Speed Regression | RMSE = %.4f | r = %.4f\n', rmse, r);

    figure;
    plot(YTest, 'k', 'DisplayName', 'True'); hold on;
    plot(YPred, 'r', 'DisplayName', 'Predicted');
    title(sprintf('Prediction vs True Speed | r = %.3f, RMSE = %.3f', r, rmse));
    legend(); xlabel('Sample'); ylabel('Speed'); grid on;
end

function [X_seqs, Y_labels] = prepare_dysco_sequences_for_regression(norm, speed, entropy, run_times, run_speeds, dysco_time, lag, seq_len)
    norm = norm(1:end-lag);
    speed = speed(1:end-lag);
    entropy = entropy(1:end-lag);
    t = dysco_time(1:end-lag);
    y_continuous = interp1(run_times, run_speeds, t, 'linear', 'extrap');
    y_continuous(y_continuous < 0) = 0;

    n = length(norm);
    X_seqs = {};
    Y_labels = [];

    for i = 1:(n - seq_len + 1)
        norm_seq = norm(i:i+seq_len-1);
        speed_seq = speed(i:i+seq_len-1);
        entropy_seq = entropy(i:i+seq_len-1);

        X_seq = [norm_seq(:), speed_seq(:), entropy_seq(:)];  % [seq_len x 3]
        target_speed = mean(y_continuous(i:i+seq_len-1));

        X_seqs{end+1,1} = X_seq;
        Y_labels(end+1,1) = target_speed;
    end
end


lag = 5;
half_window_size = 10;

net = train_cnn_lstm_regression(all_norms, all_speeds, all_entropies, ...
    running_times, running_speeds, activity_timestamps, lag, half_window_size);
