% Regresión LOGÍSTICA

%% inicialización
clear ; close all; clc

%% Cargar datos -- analizar datos!!!!

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

hold on;

% Labels
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0')
hold off;


%% =========== Parte 1: Regresión Logística Regularizada ============
%  En esta parte tenemos el inconveniente de que nuestros datos no son linearmente separables.
%
%  Por eso vamos a incluir más características -- en particular vamos a agregar características polinomiales
%

% Agregando características

X = mapeoCaracteristicas(X(:,1), X(:,2));


initial_theta = zeros(size(X, 2), 1);
lambda = 1;


[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Costo inicial (zeros): %f\n', cost);

fprintf('\nPrograma pausado. \n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%  Pueben varios valores de lambda y vean el desempeño
%
%  e.g. lambda (0, 1, 10, 100).
% ¿Qué sucede?
%

initial_theta = zeros(size(X, 2), 1);
lambda = 1;

% opciones
opciones = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, J, exit_flag] = ...
	fminunc(@(t)(funcionCostoReg(t, X, y, lambda)), initial_theta, opciones);


plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Calcular exactitud
p = predict(theta, X);

fprintf('Exactitud del entrenamiento: %f\n', mean(double(p == y)) * 100);
