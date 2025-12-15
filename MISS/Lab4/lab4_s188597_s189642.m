clear; clc; close all;
load('dataset23243.mat');

types = {
    'Gaussian', @(r, b) exp(-(b * r).^2);
    'Inverse Quadric', @(r, b) 1 ./ (1 + (b * r).^2)
};

Nt_len = length(xt);
Nv_len = length(xv);

D_tr = zeros(Nt_len, Nt_len);
for i = 1:Nt_len
    D_tr(:, i) = sqrt((xt - xt(i)).^2 + (yt - yt(i)).^2);
end

D_val = zeros(Nv_len, Nt_len);
for i = 1:Nt_len
    D_val(:, i) = sqrt((xv - xt(i)).^2 + (yv - yt(i)).^2);
end

best_mae_global = inf;
best_res_global = struct();
best_res_per_type = containers.Map('KeyType', 'char', 'ValueType', 'any');

warning('off', 'MATLAB:nearlySingularMatrix'); 
warning('off', 'MATLAB:singularMatrix');

all_mae_results = containers.Map('KeyType', 'char', 'ValueType', 'any');
betas = 0.5:0.05:3;

for i = 1:size(types, 1)
    fun = types{i, 2};
    typeName = types{i, 1};
    current_mae_list = [];
    best_mae_current_type = inf;
    best_beta_current_type = 0;
    best_w_current_type = [];

    for beta = betas
        Phi = fun(D_tr, beta);
        w = Phi \ ft;
        
        f_est = fun(D_val, beta) * w;
        mae = mean(abs(f_est - fv));
        
        current_mae_list = [current_mae_list, mae];

        if mae < best_mae_current_type
            best_mae_current_type = mae;
            best_beta_current_type = beta;
            best_w_current_type = w;
        end

        if mae < best_mae_global
            best_mae_global = mae;
            best_res_global.name = typeName;
            best_res_global.beta = beta;
            best_res_global.w = w;
            best_res_global.fun = fun;
            best_res_global.mae = mae;
        end
    end
    all_mae_results(typeName) = current_mae_list;
    best_res_per_type(typeName) = struct('beta', best_beta_current_type, 'w', best_w_current_type, 'fun', fun, 'mae', best_mae_current_type, 'name', typeName);
end

warning('on', 'MATLAB:nearlySingularMatrix');
warning('on', 'MATLAB:singularMatrix');

fprintf('Overall Best Method: %s\nBeta: %.2f\nMean Error: %.5f\n', best_res_global.name, best_res_global.beta, best_mae_global);

[X, Y] = meshgrid(0:0.1:3, -3:0.1:1);
X_col = X(:);
Y_col = Y(:);
N_grid = length(X_col);

D_grid = zeros(N_grid, Nt_len);
for j = 1:Nt_len
    D_grid(:, j) = sqrt((X_col - xt(j)).^2 + (Y_col - yt(j)).^2);
end

figure('Color', 'w');
semilogy(betas, all_mae_results('Gaussian'), 'r-o', 'LineWidth', 2);
hold on;
semilogy(betas, all_mae_results('Inverse Quadric'), 'b-o', 'LineWidth', 2);

plot(best_res_global.beta, best_res_global.mae, 'g*', 'MarkerSize', 15, 'LineWidth', 2); 

xlabel('Wartość \beta');
ylabel('Średni Błąd Bezwzględny (MAE)');
title('Błąd modelu a parametr Beta dla różnych RBF');
legend('RBF Gaussa', 'Odwrócony kwadrat RBF', 'Minimum globalne', 'Location', 'best');
grid on;
saveas(gcf, 'RBF_error_vs_beta.png');

Z_global = reshape(best_res_global.fun(D_grid, best_res_global.beta) * best_res_global.w, size(X));

figure('Color', 'w');
surf(X, Y, Z_global);
hold on;
plot3(xt, yt, ft, 'r.', 'MarkerSize', 15);
plot3(xv, yv, fv, 'b.', 'MarkerSize', 15);
xlabel('x'); ylabel('y'); zlabel('z');
title(['' best_res_global.name ', \beta: ' num2str(best_res_global.beta) ', Błąd: ' num2str(best_res_global.mae)]);
colorbar;
legend('Powierzchnia modelu', 'Zbiór A', 'Zbiór B', 'Location', 'best');
view(-45, 30);
grid on;
saveas(gcf, 'RBF_optimal_surface.png');

if strcmp(best_res_global.name, 'Gaussian')
    other_type_name = 'Inverse Quadric';
else
    other_type_name = 'Gaussian';
end

other_res = best_res_per_type(other_type_name);
Z_other = reshape(other_res.fun(D_grid, other_res.beta) * other_res.w, size(X));

figure('Color', 'w');
surf(X, Y, Z_other);
hold on;
plot3(xt, yt, ft, 'r.', 'MarkerSize', 15);
plot3(xv, yv, fv, 'b.', 'MarkerSize', 15);
xlabel('x'); ylabel('y'); zlabel('z');
title(['' other_res.name ', \beta: ' num2str(other_res.beta) ', Błąd: ' num2str(other_res.mae)]);
colorbar;
legend([ other_res.name ], 'Zbiór A', 'Zbiór B', 'Location', 'best');
view(-45, 30);
grid on;
saveas(gcf, 'RBF_second_surface.png');
