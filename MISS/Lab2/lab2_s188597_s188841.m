clc;
clear;
close all;

simulateBounces(1.0);

simulateBounces(0.8);

function simulateBounces(k_loss_factor)
    g = 10;
    m = 1;
    num_bounces = 5;

    f_surface = @(x, y) x.^2 + 2*y.^2;
    df_dx = @(x) 2*x;
    df_dy = @(y) 4*y;

    p0 = [0, 0, 200]; 
    v0 = [5, 5, 0]; 

    results = zeros(num_bounces, 7);
    trajectories = cell(num_bounces, 1);

    hFig = figure;
    hold on;
    grid on;
    
    [X, Y] = meshgrid(-10:1:10, -10:1:10);
    Z = f_surface(X, Y);
    mesh(X, Y, Z);
    
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title(['Trajektoria ruchu (k = ', num2str(k_loss_factor), ')']);
    
    view([235 55]);
    
    for i = 1:num_bounces
        
        A = v0(1)^2 + 2*v0(2)^2 + 0.5*g;
        B = 2*p0(1)*v0(1) + 4*p0(2)*v0(2) - v0(3);
        C = p0(1)^2 + 2*p0(2)^2 - p0(3);

        discriminant = B^2 - 4*A*C;
        if discriminant < 0
            error('No real intersection time found.');
        end
        
        t_all_1 = (-B + sqrt(discriminant)) / (2*A);
        t_all_2 = (-B - sqrt(discriminant)) / (2*A);

        if t_all_1 > t_all_2
            t_I = t_all_1;
        else
            t_I = t_all_2;
        end

        t_traj = linspace(0, t_I, 100);
        x_traj = p0(1) + v0(1) * t_traj;
        y_traj = p0(2) + v0(2) * t_traj;
        z_traj = p0(3) + v0(3) * t_traj - 0.5 * g * t_traj.^2;
        
        trajectories{i} = [x_traj; y_traj; z_traj];
        plot3(x_traj, y_traj, z_traj, 'r-', 'LineWidth', 2);

        xI = p0(1) + v0(1) * t_I;
        yI = p0(2) + v0(2) * t_I;
        zI = p0(3) + v0(3) * t_I - 0.5 * g * t_I^2;
        
        v_before = [v0(1), v0(2), v0(3) - g * t_I];

        E_kin = 0.5 * m * norm(v_before)^2;
        E_pot = m * g * zI;
        E_total = E_kin + E_pot;
        
        results(i, :) = [xI, yI, zI, t_I, E_kin, E_pot, E_total];
        
        N = [-df_dx(xI), -df_dy(yI), 1];
        n_vec = N / norm(N);
        
        dot_prod = dot(v_before, n_vec);
        v_after_perfect = v_before - 2 * dot_prod * n_vec;
        v_after = sqrt(k_loss_factor) * v_after_perfect;
        
        p0 = [xI, yI, zI];
        v0 = v_after;
    end
    
    hold off;

    T = array2table(results, ...
        'VariableNames', {'xI', 'yI', 'zI', 'Delta_t', 'E_kin', 'E_pot', 'E_total'});
    
    fprintf('\n\nWyniki dla k = %g:\n', k_loss_factor);
    disp(T);
    
    if k_loss_factor == 1.0
        fileName = 'Results_k1_0.csv';
        figName = 'Figure_k1_0';
    else
        fileName = 'Results_k0_8.csv';
        figName = 'Figure_k0_8';
    end

    writetable(T, fileName, 'Delimiter', ',');
    
    saveas(hFig, [figName, '.jpg']);

end
