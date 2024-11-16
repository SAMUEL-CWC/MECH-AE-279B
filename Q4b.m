% Define the parameters
alpha = 0.3;
beta = 0.3;
gamma = 0.3;

% Define the range of Hill coefficient m
m_values = 1:1:10; % You can adjust the step size for more or fewer points

% Initialize arrays to store results
num_m_values = length(m_values);
equilibria_results = zeros(num_m_values, 3); % Columns for [M, E, P]
real_eigenvalues = zeros(num_m_values, 3); % Columns for real part of eigenvalues

% Loop through each Hill coefficient value
for i = 1:num_m_values
    m = m_values(i);
    
    % Define the function to find equilibria
    equilibrium_func = @(x) [
        (1 / (1 + x(3)^m)) - alpha * x(1);  % dM/dt = 0
        x(1) - beta * x(2);                  % dE/dt = 0
        x(2) - gamma * x(3)                  % dP/dt = 0
    ];
    
    % Initial guess for [M, E, P]
    x0 = [1; 1; 1];
    
    % Solve for equilibria using fsolve
    options = optimset('Display', 'off'); % Turn off iteration display
    [x_equilibrium, ~, exitflag] = fsolve(equilibrium_func, x0, options);
    
    % Check if fsolve converged
    if exitflag > 0
        % Store equilibrium values
        equilibria_results(i, :) = x_equilibrium';
        
        % Calculate the Jacobian matrix at equilibrium
        M_eq = x_equilibrium(1);
        E_eq = x_equilibrium(2);
        P_eq = x_equilibrium(3);
        
        % Jacobian matrix elements
        J = [
            -alpha,    0,         -m * P_eq^(m-1) / (1 + P_eq^m)^2;  % Partial derivatives of dM/dt
            1,         -beta,     0;                                 % Partial derivatives of dE/dt
            0,         1,         -gamma                             % Partial derivatives of dP/dt
        ];
        
        % Find the eigenvalues of the Jacobian
        eigenvalues = eig(J);
        
        % Store the real parts of the eigenvalues
        real_eigenvalues(i, :) = real(eigenvalues)';
    else
        % If fsolve did not converge, store NaN values
        equilibria_results(i, :) = [NaN, NaN, NaN];
        real_eigenvalues(i, :) = [NaN, NaN, NaN];
    end
end

% Round the results to four decimal places
Hill_coefficient = round(m_values', 4);
M_equilibrium = round(equilibria_results(:, 1), 4);
E_equilibrium = round(equilibria_results(:, 2), 4);
P_equilibrium = round(equilibria_results(:, 3), 4);
Real_Part_Eigenvalue1 = round(real_eigenvalues(:, 1), 4);
Real_Part_Eigenvalue2 = round(real_eigenvalues(:, 2), 4);
Real_Part_Eigenvalue3 = round(real_eigenvalues(:, 3), 4);

% Create a table to store and display results
results_table = table(Hill_coefficient, M_equilibrium, E_equilibrium, P_equilibrium, ...
                      Real_Part_Eigenvalue1, Real_Part_Eigenvalue2, Real_Part_Eigenvalue3);

% Display the table
disp(results_table);
writetable(results_table, 'equilibria_results.csv');

% Initial conditions for phase space plots
initial_conditions = [
    0.5, 0.5, 0.5;
    1.0, 1.0, 1.0;
    1.5, 1.5, 1.5;
    2.0, 2.0, 2.0;
    2.5, 2.5, 2.5
];

% Plot for m = 2
m = 2;
subplot(1, 2, 1);
hold on;
for i = 1:size(initial_conditions, 1)
    [t, y] = ode45(@(t, x) [
        (1 / (1 + x(3)^m)) - alpha * x(1);
        x(1) - beta * x(2);
        x(2) - gamma * x(3)
    ], [0 50], initial_conditions(i, :)');
    plot(y(:, 2), y(:, 3));
end
% Plot equilibrium point as a star
plot(E_equilibrium(m_values == m), P_equilibrium(m_values == m), ...
    'pentagram', 'MarkerFaceColor','red', 'MarkerSize', 10);
xlabel('E', 'FontSize', 12);
ylabel('P', 'FontSize', 12);
title('Phase Space (E vs P) for m = 2', 'FontSize', 16);
grid on;
hold off;

% Plot for m = 10
m = 10;
subplot(1, 2, 2);
hold on;
for i = 1:size(initial_conditions, 1)
    [t, y] = ode45(@(t, x) [
        (1 / (1 + x(3)^m)) - alpha * x(1);
        x(1) - beta * x(2);
        x(2) - gamma * x(3)
    ], [0 50], initial_conditions(i, :)');
    plot(y(:, 2), y(:, 3));
end
% Plot equilibrium point as a star
plot(E_equilibrium(m_values == m), P_equilibrium(m_values == m), ...
    'pentagram', 'MarkerFaceColor','red', 'MarkerSize', 10);
xlabel('E', 'FontSize', 12);
ylabel('P', 'FontSize', 12);
title('Phase Space (E vs P) for m = 10', 'FontSize', 16);
grid on;
hold off;
saveas(gcf, 'Q4b.png');