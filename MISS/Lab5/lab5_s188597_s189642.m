clear; clc; close all;

f = @(x,y) x.^2 + 2.*x.*y;
V_analytic = 0.5;

x_max = 1;
y_max = 1;
area = 0.5;
h = 5;

main_i = 4:7;
N_vec = zeros(length(main_i),1);
V_err_vec = zeros(length(main_i),1);
C_err_vec = zeros(length(main_i),1);

fprintf('%-10s %-15s %-15s %-15s %-15s\n', 'N', 'Hit & Miss', 'Err H&M', 'Mean Value', 'Err MV');

for j = 1:length(main_i)
    N = 10^main_i(j);
    N_vec(j) = N;
    
    x_final = zeros(N, 1);
    y_final = zeros(N, 1);
    i = 1;
    
    while i <= N
        tx = rand * x_max;
        ty = rand * y_max;
        if ty < tx
            x_final(i) = tx;
            y_final(i) = ty;
            i = i + 1;
        end
    end

    z_vec = rand(N, 1) * h;
    values_vec = f(x_final, y_final);

    hits = sum(z_vec < values_vec);
    V = (area * h) * (hits / N);
    C = area * mean(values_vec);

    V_err_vec(j) = abs(V - V_analytic);
    C_err_vec(j) = abs(C - V_analytic);

    fprintf('%-10d %-15.5f %-15.5f %-15.5f %-15.5f\n', length(values_vec), V, V_err_vec(j), C, C_err_vec(j));
end

figure;
loglog(N_vec, V_err_vec, 'r-o', 'LineWidth', 1.5);
hold on;
loglog(N_vec, C_err_vec, 'b-x', 'LineWidth', 1.5);
grid on;
xlabel('Liczba prób (N)');
ylabel('Błąd bezwzględny');
title('Zbieżność metod Monte Carlo');
legend('Metoda Hit and Miss', 'Metoda Wartości Średniej');
