i=8;
dysco_time = activity_timestamps{i}(half_window_size+1:end-half_window_size);
y = interp1(running_times{i}, running_speeds{i}, dysco_time(1:end-lag), 'linear', 'extrap');
% Create figure
figure;
set(gcf, 'Position', [100 100 1000 1000]);

%% 1. DySCo Norm
subplot(6,1,1)
plot(dysco_time, all_norms{i}, 'k');
hold on;
cp_norm = findchangepts(all_norms{i}, 'Statistic', 'mean', 'MaxNumChanges', 10);
xline(dysco_time(cp_norm), 'r--', 'LineWidth', 1.5);
title('DySCo Norm with Change Points');
ylabel('Norm');
xlim([dysco_time(1), dysco_time(end)]);

%% 2. Von Neumann Entropy
subplot(6,1,2)
plot(dysco_time, all_entropies{i}, 'm');
title('Von Neumann Entropy');
ylabel('Entropy');
xlim([dysco_time(1), dysco_time(end)]);

%% 3. Functional Connectivity Dynamics (FCD)
subplot(6,1,3)
imagesc(dysco_time, dysco_time, all_FCDs{i});
axis xy;
colorbar;
title('Functional Connectivity Dynamics (FCD)');
xlabel('Time');
ylabel('Time');

%% 4. Reconfiguration Speed with Change Points
len_speed = min(length(dysco_time), length(all_speeds{i}));
dysco_time_speed = dysco_time(1:len_speed);
speed_vals = all_speeds{i}(1:len_speed);

subplot(6,1,4)
plot(dysco_time_speed, speed_vals, 'r');
hold on;
cp_speed = findchangepts(all_speeds{i}, 'Statistic', 'mean', 'MaxNumChanges', 10);
xline(dysco_time(cp_speed), 'b--', 'LineWidth', 1.5);
title('Reconfiguration Speed with Change Points');
ylabel('Speed');
xlabel(['Time (lag = ' num2str(lag) ')']);
xlim([dysco_time(1), dysco_time(end)]);

%% 5. Running Speed
subplot(6,1,5)
plot(running_times{i}, running_speeds{i}, 'Color', [0.1216, 0.4667, 0.7059], 'LineWidth', 2);
ylabel('Running Speed');
xlabel('Time');
xlim([dysco_time(1), dysco_time(end)]);
set(gca, 'YColor', [0.1216, 0.4667, 0.7059]);
title('Running Speed');

%% 6. Eye Diameter
subplot(6,1,6)
plot(eye_timestamps{i}, eye{i}, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5);
ylabel('Eye Diameter');
xlabel('Time');
xlim([dysco_time(1), dysco_time(end)]);
title('Eye Tracking');

sgtitle('DySCo Dynamics and Behavioral Metrics with Change Points');
