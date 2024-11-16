%Interconnected Genes
% Parameters
alpha_m = 1;
beta_m = 0.1;
alpha_p = 2;
beta_p = 0.05;
KD = 200;
Y = 100;

% Time span for simulation
tspan = [0 200];

% Initial conditions for M1, M2, X, P
ICs = [0, 0, 0, 0];

% Solve ODEs in case 1
[t1, sol1] = ode45(@(t, y) geneODEs1(t, y, alpha_m, beta_m, alpha_p, beta_p, KD, Y), tspan, ICs);

% Solve ODEs in case 2
[t2, sol2] = ode45(@(t, y) geneODEs2(t, y, alpha_m, beta_m, alpha_p, beta_p, KD, Y), tspan, ICs);

% Plot the results for Function 1
figure;
plot(t1, sol1(:, 1), 'LineWidth', 2, 'DisplayName', 'M_{1}');
hold on;
plot(t1, sol1(:, 2), 'LineWidth', 2, 'DisplayName', 'M_{2}');
plot(t1, sol1(:, 3), 'LineWidth', 2, 'DisplayName', 'X');
plot(t1, sol1(:, 4), 'LineWidth', 2, 'DisplayName', 'P');
hold off;
xlabel('Time');
ylabel('Concentration');
legend('Location', 'best');
title('Normalized Dynamics in Case a');
saveas(gcf, 'Q2a.png');

% Plot the results for Function 2
figure;
plot(t2, sol2(:, 1), 'LineWidth', 2, 'DisplayName', 'M_{1}');
hold on;
plot(t2, sol2(:, 2), 'LineWidth', 2, 'DisplayName', 'M_{2}');
plot(t2, sol2(:, 3), 'LineWidth', 2, 'DisplayName', 'X');
plot(t2, sol2(:, 4), 'LineWidth', 2, 'DisplayName', 'P');
hold off;
xlabel('Time');
ylabel('Concentration');
legend('Location', 'best');
title('Normalized Dynamics in Case b');
saveas(gcf, 'Q2b.png');

% ODE function for Case 1
function dydt = geneODEs1(~, y, alpha_m, beta_m, alpha_p, beta_p, KD, Y)
    % Variables
    M1 = y(1);
    M2 = y(2);
    X = y(3);
    P = y(4);
    
    % APF calculations
    APF_1 = Y / (KD + Y);
    APF_2a = (Y / KD) / ((1 + X / KD) * (1 + Y / KD));
    
    % ODEs
    dM1_dt = alpha_m * APF_1 - beta_m * M1;
    dM2_dt = alpha_m * APF_2a - beta_m * M2;
    dX_dt = alpha_p * M1 - beta_p * X;
    dP_dt = alpha_p * M2 - beta_p * P;
    
    % Return derivatives
    dydt = [dM1_dt; dM2_dt; dX_dt; dP_dt];
end

% ODE function for Case 2
function dydt = geneODEs2(~, y, alpha_m, beta_m, alpha_p, beta_p, KD, Y)
    % Variables
    M1 = y(1);
    M2 = y(2);
    X = y(3);
    P = y(4);
    
    % APF calculations
    APF_1 = Y / (KD + Y);
    APF_2b = (Y / KD) / (1 + Y / KD + X / KD);
    
    % ODEs
    dM1_dt = alpha_m * APF_1 - beta_m * M1;
    dM2_dt = alpha_m * APF_2b - beta_m * M2;
    dX_dt = alpha_p * M1 - beta_p * X;
    dP_dt = alpha_p * M2 - beta_p * P;
    
    % Return derivatives
    dydt = [dM1_dt; dM2_dt; dX_dt; dP_dt];
end
