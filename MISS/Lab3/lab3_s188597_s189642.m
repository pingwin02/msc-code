clear; clc; close all;
mu = 0.01;
T = 100;
L = 1;
M = 100;
N = 2000;
b_task1 = 0;
b_task2 = 0.5;

x = linspace(0, L, M)';
dx = x(2) - x(1);
v = sqrt(T / mu);
dt = dx / v;
t = (0:N-1)' * dt;

s_fun = @(x) (sin(5*x) + 1) ./ (x + 1);
g_fun = @(x) 4 * (x - 0.5).^2 - 1;

s_vec = s_fun(x);
g_vec = g_fun(x);
l = 1;
k = 0.7;

f_nodamp = simulate_wave_equation(mu, T, L, M, N, b_task1, dt, s_vec, g_vec, l, k);

f_damped = simulate_wave_equation(mu, T, L, M, N, b_task2, dt, s_vec, g_vec, l, k);

c = 0.7;
d = 1;
y_eq = c * x + d;
fprintf('y_E(x) = %.2fx + %.2f\n', c, d);

fig1 = figure();
[T_mesh, X_mesh] = meshgrid(t, x);
surf(T_mesh, X_mesh, f_nodamp);
shading interp;
xlabel('Czas t (s)');
ylabel('Pozycja x (m)');
zlabel('Wychylenie y (m)');
title('Wibracje bez tłumienia (b=0)');
view(30, 30);
saveas(fig1, 'task1_no_damping.png');

fig2 = figure();
surf(T_mesh, X_mesh, f_damped);
shading interp;
xlabel('Czas t (s)');
ylabel('Pozycja x (m)');
zlabel('Wychylenie y (m)');
title('Wibracje z tłumieniem (b=0.5)');
view(30, 30);
saveas(fig2, 'task2_with_damping.png');

fig3 = figure();
plot(x, f_damped(:, N), 'b-', 'LineWidth', 2, 'DisplayName', 'Ostatnia iteracja (n=2000)');
hold on;
plot(x, y_eq, 'r--', 'LineWidth', 2, 'DisplayName', 'Rozwiązanie równowagowe (y_E)');
xlabel('Pozycja x (m)');
ylabel('Wychylenie y (m)');
title('Porównanie ostatniej iteracji z rozwiązaniem równowagowym');
legend('show', 'Location', 'best');
grid on;
saveas(fig3, 'task3_equilibrium_solution.png');


function f = simulate_wave_equation(mu, T, L, M, N, b, dt, s_vec, g_vec, l, k)
    v = sqrt(T / mu);
    dx = L / (M - 1);
    beta = b / (2 * mu);
    p = (v * dt / dx)^2;
    q = 1 + beta * dt;
    u = 1 - beta * dt;
    f = zeros(M, N);

    f(:, 1) = s_vec;
    f(1, :) = l;
    
    f(2:M-1, 2) = p/2 * (f(3:M, 1) - 2*f(2:M-1, 1) + f(1:M-2, 1)) + f(2:M-1, 1) + u * dt * g_vec(2:M-1);
    
    f(M, 2) = p/2 * (f(M-1, 1) + 2*dx*k - 2*f(M, 1) + f(M-1, 1)) + f(M, 1) + u * dt * g_vec(M);
    
    for n = 2:N-1
        f(2:M-1, n+1) = p/q * (f(3:M, n) - 2*f(2:M-1, n) + f(1:M-2, n)) + 2/q * f(2:M-1, n) - u/q * f(2:M-1, n-1);
        
        f(M, n+1) = p/q * (f(M-1, n) + 2*dx*k - 2*f(M, n) + f(M-1, n)) + 2/q * f(M, n) - u/q * f(M, n-1);
    end
end
