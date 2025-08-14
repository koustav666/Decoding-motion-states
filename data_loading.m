% List of data directories
paths = {
    '..\notebooks\SS038\2015-02-17\001'
    '..\notebooks\SS041\2015-04-23\001'
    '..\notebooks\SS044\2015-04-28\001'
    '..\notebooks\SS044\2015-05-29\001'
    '..\notebooks\SS045\2015-05-04\001'
    '..\notebooks\SS045\2015-05-05\001'
    '..\notebooks\SS047\2015-11-23\001'
    '..\notebooks\SS047\2015-12-03\001'
    '..\notebooks\SS048\2015-11-09\001'
    '..\notebooks\SS048\2015-12-02\001'
};

activity_data = cell(length(paths), 1);
activity_timestamps = cell(length(paths), 1);
running_times = cell(length(paths), 1);
running_speeds = cell(length(paths), 1);
eye = cell(length(paths), 1);
eye_timestamps = cell(length(paths), 1);
nROI_pos = cell(length(paths), 1);
% Loop through each path and load data
for i = 1:length(paths)
    path = paths{i};
    dff_file = fullfile(path, '_ss_2pCalcium.dff.npy');
    ts_file = fullfile(path, '_ss_2pCalcium.timestamps.npy');
    interval_file = fullfile(path, '_ss_recordings.grayScreen_intervals.npy');
    running_times_file = fullfile(path, '_ss_running.timestamps.npy');
    running_speeds_file = fullfile(path, '_ss_running.speed.npy');
    eye_timestamps_file = fullfile(path, 'eye.timestamps.npy'); 
    eye_file = fullfile(path, 'eye.diameter.npy');
    nROI_pos_file = fullfile(path, '_ss_2pRois.xyz.npy');

    % File existence check
    if ~isfile(dff_file) || ~isfile(ts_file) || ~isfile(interval_file) || ...
       ~isfile(running_times_file) || ~isfile(running_speeds_file)
        fprintf('Skipping: %s (missing files)\n', path);
        continue;
    end

    % Load data
    dff = readNPY(dff_file);
    timestamps = readNPY(ts_file);
    intervals = readNPY(interval_file);
    
    % Interval check
    if isempty(intervals) || size(intervals, 2) < 2
        fprintf('Skipping: %s (invalid interval)\n', path);
        continue;
    end

    interval = intervals(1, :);
    ind_time = find(timestamps > interval(1) & timestamps < interval(2));
    if isempty(ind_time)
        fprintf('Skipping: %s (no data in interval)\n', path);
        continue;
    end

    % Load and store
    running_times{i} = readNPY(running_times_file);
    running_speeds{i} = readNPY(running_speeds_file);
    activity_data{i} = dff(ind_time, :);
    activity_timestamps{i} = timestamps(ind_time);
    eye{i} = readNPY(eye_file);
    eye_timestamps{i} = readNPY(eye_timestamps_file);
    nROI_pos{i} = readNPY(nROI_pos_file);
    fprintf('Loaded: %s (%d timepoints)\n', path, length(ind_time));
end