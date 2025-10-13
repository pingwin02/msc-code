clear; clc; close all;

g = 10;
N = 100000;
tolerance = 0.05;
v0_min = 1; v0_max = 30;
alpha_min = 1; alpha_max = 89;

valid_v0 = [];
valid_alpha = [];
valid_trajectories = {};

for i = 1:N
    v0 = v0_min + rand()*(v0_max - v0_min);
    alpha = alpha_min + rand()*(alpha_max - alpha_min);
    alpha_rad = deg2rad(alpha);
    
    T = 2*v0*sin(alpha_rad)/g;
    t = linspace(0, T, 200);
    
    x = v0*cos(alpha_rad) .* t;
    y = v0*sin(alpha_rad) .* t - 0.5*g*t.^2;
    
    hit_R1 = any(abs(y - 3) < tolerance & x >= 1 & x <= 5);
    hit_R2 = any(abs(y - 3) < tolerance & x >= 9 & x <= 13);
    hit_R3 = any(abs(x - 7) < tolerance & y >= 4 & y <= 5);
    hit_R4 = any(abs(x - 7) < tolerance & y >= 7 & y <= 8);
    
    if hit_R1 && hit_R2 && ~hit_R3 && ~hit_R4
        valid_v0(end+1) = v0;
        valid_alpha(end+1) = alpha;
        if length(valid_trajectories) < 5
            valid_trajectories{end+1} = [x; y];
        end
    end
end

figure;
scatter(valid_alpha, valid_v0, '.', 'g');
xlabel('\alpha [stopnie]');
ylabel('v_0 [m/s]');
title('Poprawne kombinacje (\alpha, v_0)');
grid on;
saveas(gcf,'valid_alpha_v0.png');

figure; hold on;
plot([1 5], [3 3], 'g', 'LineWidth', 5);
plot([9 13], [3 3], 'g', 'LineWidth', 5);
plot([7 7], [4 5], 'r', 'LineWidth', 5);
plot([7 7], [7 8], 'r', 'LineWidth', 5);

for k = 1:length(valid_trajectories)
    traj = valid_trajectories{k};
    plot(traj(1,:), traj(2,:), '-');
end

xlabel('x');
ylabel('y');
title('Trajektorie');
legend('R1','R2','R3', 'R4','Trajektorie');
grid on;
ylim([0 10]);
xlim([0 14]);
saveas(gcf,'valid_trajectories.png');

probability = length(valid_v0) / N;
fprintf('Prawdopodobieństwo: %.3f%s\n', probability * 100, '%');
