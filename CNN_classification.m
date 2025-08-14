train_rats = setdiff(1:length(all_norms), [1, 4]);;
test_rat = 3;
half_window_size = 10;
max_train_samples = 1500; % limit for training samples

%% ---------------- Train Data ----------------
XTrain_all = [];
YTrain_all = [];
inputSize = size(avg_mini_truncated_matrix{train_rats(1)}(:,:,1)); % base size

for rat = train_rats
    t_train_full = activity_timestamps{rat}(half_window_size+1:end-half_window_size);
    speed_train_full = interp1(running_times{rat}, running_speeds{rat}, t_train_full, 'linear', 'extrap');

    % --- Subsample to max_train_samples ---
    if length(t_train_full) > max_train_samples
        idx_sel = randperm(length(t_train_full), max_train_samples);
        idx_sel = sort(idx_sel); % keep chronological order
    else
        idx_sel = 1:length(t_train_full);
    end

    t_train = t_train_full(idx_sel);
    speed_train = speed_train_full(idx_sel);

    labels_train_cat = categorical(speed_train > 1, [false true], {'stationary', 'running'});

    numSamples = length(t_train);
    Xtemp = zeros([inputSize, 1, numSamples], 'single');

    for n = 1:numSamples
        img = single(avg_mini_truncated_matrix{rat}(:,:,idx_sel(n)));
        img = imresize(img, [inputSize(1) inputSize(2)]);
        range_val = max(img(:)) - min(img(:));
        if range_val > 0
            img = (img - min(img(:))) / (range_val + eps);
        else
            img(:) = 0;
        end
        Xtemp(:,:,1,n) = img;
    end

    % Append to training data
    if isempty(XTrain_all)
        XTrain_all = Xtemp;
        YTrain_all = labels_train_cat;
    else
        XTrain_all = cat(4, XTrain_all, Xtemp);
        YTrain_all = [YTrain_all; labels_train_cat];
    end
end

%% ---------------- Test Data (Rat 3) ----------------
t_test = activity_timestamps{test_rat}(half_window_size+1:end-half_window_size);
speed_test = interp1(running_times{test_rat}, running_speeds{test_rat}, t_test, 'linear', 'extrap');
labels_test_cat = categorical(speed_test > 1, [false true], {'stationary', 'running'});

numSamples_test = length(t_test);
XTest = zeros([inputSize, 1, numSamples_test], 'single');

for n = 1:numSamples_test
    img = single(avg_mini_truncated_matrix{test_rat}(:,:,n));
    img = imresize(img, [inputSize(1) inputSize(2)]);
    range_val = max(img(:)) - min(img(:));
    if range_val > 0
        img = (img - min(img(:))) / (range_val + eps);
    else
        img(:) = 0;
    end
    XTest(:,:,1,n) = img;
end

%% ---------------- CNN Layers ----------------
layers = [
    imageInputLayer([inputSize 1], 'Name', 'input')

    convolution2dLayer(3, 8, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(2)
    softmaxLayer()
    classificationLayer()
];

%% ---------------- Training Options ----------------
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% ---------------- Train & Test ----------------
net = trainNetwork(XTrain_all, YTrain_all, layers, options);

YPred = classify(net, XTest);

%% ---------------- Metrics ----------------
accuracy = mean(YPred == labels_test_cat);
fprintf('Test Accuracy (Rat %d): %.2f%%\n', test_rat, accuracy * 100);

tp = sum((labels_test_cat == 'running') & (YPred == 'running'));
fp = sum((labels_test_cat == 'stationary') & (YPred == 'running'));
fn = sum((labels_test_cat == 'running') & (YPred == 'stationary'));
prec = tp / (tp + fp + eps);
rec = tp / (tp + fn + eps);
f1 = 2 * (prec * rec) / (prec + rec + eps);

fprintf('Precision: %.4f\n', prec);
fprintf('Recall: %.4f\n', rec);
fprintf('F1 Score: %.4f\n', f1);

%% ---------------- Confusion Matrix ----------------
figure;
confusionchart(labels_test_cat, YPred);
title(sprintf('Rat %d Classification Results (CNN)', test_rat));
