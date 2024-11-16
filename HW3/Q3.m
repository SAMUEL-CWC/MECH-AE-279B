% Repressilator Model Simulation
% Parameters
alpha = 250;
alpha0 = 0;
n = 2;
beta = 5;

% Time span for simulation
tspan = [0 60]; % Simulate from time 0 to 100

% Initial conditions for m1, m2, m3, p1, p2, p3
ICs = rand(1,6); % Arbitrary non-zero initial values

% Solve ODE using ode45
[t, y] = ode45(@(t, y) repressilator_ode(t, y, alpha, alpha0, n, beta), tspan, ICs);

% Extract solutions
m1 = y(:, 1);
m2 = y(:, 2);
m3 = y(:, 3);
p1 = y(:, 4);
p2 = y(:, 5);
p3 = y(:, 6);

% Plot oscillatory solutions
figure;

% Plot mRNA levels
subplot(2, 1, 1);
plot(t, m1, 'DisplayName', 'm_{1}');
hold on;
plot(t, m2, 'DisplayName', 'm_{2}');
plot(t, m3, 'DisplayName', 'm_{3}');
xlabel('Time', 'FontSize', 12);
ylabel('Concentration', 'FontSize', 12);
title('Repressilator mRNA Dynamics', 'FontSize', 16);
legend('Location', 'best');
grid on;

% Plot protein levels
subplot(2, 1, 2);
plot(t, p1, 'DisplayName', 'p_{1}');
hold on;
plot(t, p2, 'DisplayName', 'p_{2}');
plot(t, p3, 'DisplayName', 'p_{3}');
xlabel('Time', 'FontSize', 12);
ylabel('Concentration', 'FontSize', 12);
title('Repressilator Protein Dynamics', 'FontSize', 16);
legend('Location', 'best');
grid on;

saveas(gcf, 'Q3a.png');

% Phase space plot
figure;
subplot(3,1,1)
plot(m1, p1);
xlabel('m_{1}', 'FontSize', 12);
ylabel('p_{1}', 'FontSize', 12);
grid on;
title('Phase Space (m_{1} vs p_{1})', 'FontSize', 16);

subplot(3,1,2)
plot(m2, p2);
xlabel('m_{2}', 'FontSize', 12);
ylabel('p_{2}', 'FontSize', 12);
grid on;
title('Phase Space (m_{2} vs p_{2})', 'FontSize', 16);

subplot(3,1,3)
plot(m3, p3);
xlabel('m_{3}', 'FontSize', 12);
ylabel('p_{3}', 'FontSize', 12);
grid on;
title('Phase Space (m_{3} vs p_{3})', 'FontSize', 16);

saveas(gcf, 'Q3b.png');

% Function defining the system of ODEs
function dydt = repressilator_ode(~, y, alpha, alpha0, n, beta)
    % Variables
    m1 = y(1);
    m2 = y(2);
    m3 = y(3);
    p1 = y(4);
    p2 = y(5);
    p3 = y(6);
    
    % ODEs based on given equations
    dm1_dt = -m1 + alpha / (1 + p3^n) + alpha0;
    dm2_dt = -m2 + alpha / (1 + p1^n) + alpha0;
    dm3_dt = -m3 + alpha / (1 + p2^n) + alpha0;
    
    dp1_dt = beta * (m1 - p1);
    dp2_dt = beta * (m2 - p2);
    dp3_dt = beta * (m3 - p3);
    
    % Return derivatives
    dydt = [dm1_dt; dm2_dt; dm3_dt; dp1_dt; dp2_dt; dp3_dt];
end
