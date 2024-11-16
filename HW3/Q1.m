% Define ranges for Y' and X' (normalized Y and X)
Y_prime = linspace(0, 15, 100);
X_prime = linspace(0, 15, 100);
[Y, X] = meshgrid(Y_prime, X_prime);

% Independent binding sites
P_active_independent = Y ./ ((1 + X) .* (1+ Y));

% Overlapping binding sites
P_active_overlapping = Y ./ (1 + X + Y);

% Plotting for independent binding sites
figure;
surf(Y, X, P_active_independent);
title('APF of case a (Independent Binding Sites)', 'FontSize', 16);
xlabel('Y/K_{DY}', 'FontSize', 12);
ylabel('X/K_{DX}', 'FontSize', 12);
zlabel('APF', 'FontSize', 12);
shading interp;
grid on;
saveas(gcf, 'Q1a.png');

% Plotting for overlapping binding sites
figure;
surf(Y, X, P_active_overlapping);
title('APF of case b (Overlapping Binding Sites)', 'FontSize', 16);
xlabel('Y/K_{DY}', 'FontSize', 12);
ylabel('X/K_{DX}', 'FontSize', 12);
zlabel('APF', 'FontSize', 12);
shading interp;
grid on;
saveas(gcf, 'Q1b.png');
