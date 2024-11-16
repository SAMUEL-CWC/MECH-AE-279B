% Define the parameters
alpha = 0.3;
beta = 0.3;
gamma = 0.3;

% Define the Hill coefficient
m = 2;

% Define the function to find equilibria
equilibrium_func = @(x) [
    (1 / (1 + x(3)^m)) - alpha * x(1);   % dM/dt = 0
    x(1) - beta * x(2);                  % dE/dt = 0
    x(2) - gamma * x(3)                  % dP/dt = 0
];

% Initial guess for [M, E, P]
x0 = [1; 1; 1];

% Set options for fsolve using optimset
options = optimset('Display', 'iter'); % Display iteration steps

% Solve for equilibria using fsolve
[x_equilibrium, fval, exitflag] = fsolve(equilibrium_func, x0, options);

% Display the results
if exitflag > 0
    fprintf('Equilibrium values found:\n');
    fprintf('M = %f\n', x_equilibrium(1));
    fprintf('E = %f\n', x_equilibrium(2));
    fprintf('P = %f\n', x_equilibrium(3));
else
    fprintf('Failed to converge to an equilibrium.\n');
end

