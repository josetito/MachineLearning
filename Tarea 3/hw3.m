%% Initialization
clear ; close all; clc

%% ==================== Secci'on 1: Funci'on B'asica ====================
% Complete calentamiento.m 
fprintf('Corriendo calentamiento... \n');
fprintf('5x5 matriz identidad: \n');
calentamiento()

fprintf('Pausa, presione enter para continuar.\n');
pause;



%% ======================= Secci'on 2: Graficando =======================
fprintf('Graficando los datos ...\n')
data = load('hw3data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % n'umero de training examples

% Graficar
% completar c'odigo en graficar.m
graficar(X, y);

fprintf('Pausa, presiones enter para continuar.\n');
pause;
%% =================== Sección 3: Gradiente descendente ===================
fprintf('Corriendo Gradiente descendente ...\n')

X = [ones(m, 1), data(:,1)]; % agregar una columna
theta = zeros(2, 1); % inicializar parámetros

% opciones
iterations = 1500;%iteraciones
alpha = 0.01;

% costo inicial
costo(X, y, theta)

% correr gradiente descendente
theta = gradienteDescendente(X, y, theta, alpha, iterations)

% imprimir los valores encontrados
fprintf('Theta: ');
fprintf('%f %f \n', theta(1), theta(2));

% graficar
hold on; % mantener el gráfico anterior visible para ver la línea (hipótesis)
plot(X(:,2), X*theta, '-')
legend('Training data', 'Regresión lineal')
hold off % 

% Predecir valores para diferentes poblaciones: 35,000 y 70,000
predict1 = [1, 3.5] *theta;
fprintf('Para una población = 35,000, predecimos una ganancia de %f\n', predict1*10000);
predict2 = [1, 7] * theta;
fprintf('Para una población = 70,000, predecimos una ganancia de %f\n', predict2*10000);

fprintf('Pausa, presiones enter para continuar.\n');

%% ============= sección 4: Visualizando J(theta_0, theta_1) =============
fprintf('Visualizando J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% inicializar los valores J_vals en una matriz de 0s
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% llenar J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Para graficar
J_vals = J_vals';
% gráfico de superficie
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% gráfico de contorno
figure;

contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
pause;
