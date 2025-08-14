% Parameters
fs = 7.5;
rat_no = 8;
band_lo = 0;
band_hi = 0.5;

% Load data
ssignal = activity_data{rat_no};  % [time x channels]
t = activity_timestamps{rat_no};
interp_running_speed = interp1(running_times{rat_no}, running_speeds{rat_no}, t, 'linear', 'extrap');

num_channels = size(ssignal, 2);
correlations = zeros(num_channels, 1);
pvals = zeros(num_channels, 1);

% Compute band power and Kendall correlation
for col = 1:num_channels
    signal = ssignal(:, col);
    [cfs, freq] = cwt(signal, 'amor', fs);
    power = abs(cfs).^2;
    
    band_idx = (freq >= band_lo) & (freq <= band_hi);
    band_power = mean(power(band_idx, :), 1);
    
    [rho, p] = corr(band_power(:), interp_running_speed(:), 'Type', 'Kendall');
    correlations(col) = rho;
    pvals(col) = p;
end

% ROI positions
xyz = nROI_pos{rat_no}(:, 1:3);

% Sort channels by absolute correlation
[~, sortedIdx] = sort(abs(correlations), 'descend');
topN = 5;
topChannels = sortedIdx(1:topN);

%% UI Figure with Interactive Heatmap
f = uifigure('Name', 'Interactive 3D Correlation Heatmap', 'Position', [100 100 900 600]);

% Axes
ax = uiaxes(f, 'Position', [50 100 600 450]);
ax.XLabel.String = 'X';
ax.YLabel.String = 'Y';
ax.ZLabel.String = 'Z';
ax.Title.String = sprintf('ROIs with Kendall \\tau ≥ %.2f', 0);

% Initial scatter
mask = correlations >= 0;
sc = scatter3(ax, xyz(mask,1), xyz(mask,2), xyz(mask,3), ...
    60, correlations(mask), 'filled');
colorbar(ax);
colormap(ax, jet);
clim(ax, [-1 1]);
grid(ax, 'on');
view(ax, 3);

% Slider for correlation threshold
sld = uislider(f, ...
    'Position', [150 50 600 3], ...
    'Limits', [-1 1], ...
    'Value', 0, ...
    'MajorTicks', -1:0.2:1, ...
    'ValueChangedFcn', @(src, event) updatePlot(ax, sc, xyz, correlations, src.Value));

% Text box to show top correlated channels
txt = uitextarea(f, 'Position', [670 100 200 450], ...
    'Editable', 'off', ...
    'FontName', 'Courier New', ...
    'FontSize', 12);

% Display top channels
txt.Value = formatTopChannels(topChannels, correlations, pvals);

%% Helper function: update plot
function updatePlot(ax, sc, xyz, correlations, thresh)
    mask = correlations >= thresh;
    if any(mask)
        sc.XData = xyz(mask,1);
        sc.YData = xyz(mask,2);
        sc.ZData = xyz(mask,3);
        sc.CData = correlations(mask);
        sc.SizeData = 60;
    else
        sc.XData = NaN; sc.YData = NaN; sc.ZData = NaN;
        sc.CData = NaN;
    end
    ax.Title.String = sprintf('ROIs with Kendall \\tau ≥ %.2f', thresh);
end

%% Helper function: format top correlations
function str = formatTopChannels(topIdx, correlations, pvals)
    str = "Top 5 ROIs (by |Kendall \tau|):";
    for i = 1:length(topIdx)
        idx = topIdx(i);
        line = sprintf('ROI %3d:  \tau = %+0.3f,  p = %.3g', idx, correlations(idx), pvals(idx));
        str(end+1) = line; %#ok<AGROW>
    end
end

best_ch = topChannels(1);

% Get signal and compute band power
signal = ssignal(:, best_ch);
[cfs, freq] = cwt(signal, 'amor', fs);
power = abs(cfs).^2;
band_idx = (freq >= band_lo) & (freq <= band_hi);
band_power = mean(power(band_idx, :), 1);

% Normalize both for comparison
norm_power = (band_power - mean(band_power)) / std(band_power);
norm_speed = (interp_running_speed - mean(interp_running_speed)) / std(interp_running_speed);

% Plot
%% Plot band power of Top 5 ROIs vs Running Speed

figure('Name', 'Top 5 ROIs: Band Power vs Running Speed', 'Position', [100 100 1000 800]);

for i = 1:topN
    ch = topChannels(i);
    signal = ssignal(:, ch);

    % Wavelet and band power
    [cfs, freq] = cwt(signal, 'amor', fs);
    power = abs(cfs).^2;
    band_idx = (freq >= band_lo) & (freq <= band_hi);
    band_power = mean(power(band_idx, :), 1);

    % Normalize
    norm_power = (band_power - mean(band_power)) / std(band_power);
    norm_speed = (interp_running_speed - mean(interp_running_speed)) / std(interp_running_speed);

    % Plot each in a subplot
    subplot(topN, 1, i);
    plot(t, norm_power, 'r-', 'LineWidth', 1.2); hold on;
    plot(t, norm_speed, 'b--', 'LineWidth', 1.2);
    ylabel(sprintf('ROI #%d', ch));
    title(sprintf('Kendall \\tau = %.2f, p = %.3g', correlations(ch), pvals(ch)));
    if i == topN
        xlabel('Time (s)');
    end
    if i == 1
        legend('Band Power (0–0.5 Hz)', 'Running Speed');
    end
    grid on;
end

sgtitle('Top 5 Correlated ROIs: Band Power vs Running Speed (Z-scored)');
