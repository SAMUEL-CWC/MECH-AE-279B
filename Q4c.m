% Define the parameters
alpha = 0.3;
beta = 0.3;
gamma = 0.3;

% Define the range for Hill coefficient m
m_values = 1:1:12;

% Initialize arrays to store the minimum and maximum values of P
P_min = zeros(length(m_values), 1);
P_max = zeros(length(m_values), 1);

% Set the initial conditions for [M, E, P]
initial_conditions = rand(1,3);

% Loop through each Hill coefficient value
for i = 1:length(m_values)
    m = m_values(i);
    
    % Define the ODE system
    ode_func = @(t, x) [
        (1 / (1 + x(3)^m)) - alpha * x(1);
        x(1) - beta * x(2);
        x(2) - gamma * x(3)
    ];
    
    % Set the time span for simulation
    t_span = [0 10000];
    
    % Solve the ODE for the initial condition
    [t, y] = ode45(ode_func, t_span, initial_conditions');
    
    % Store the minimum and maximum values of P
    P_min(i) = min(y(:, 3));
    P_max(i) = max(y(:, 3));
end

% Create a bifurcation diagram
figure;
hold on;
plot(m_values, P_min, 'o-', 'MarkerSize', 5, 'DisplayName', 'Min P');
plot(m_values, P_max, 'o-', 'MarkerSize', 5, 'DisplayName', 'Max P');
xlabel('Hill Coefficient (m)');
ylabel('P (Min/Max)');
title('Bifurcation Diagram of P vs m');
legend('show');
grid on;
hold off;

saveas(gcf, 'Q4c.png');