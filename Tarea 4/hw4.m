% REGRESIÓN LOGÍSTICA
% Archivos a modificar:
%     sigmoid.m
%     funcionCostoReg.m
%     predict.m
%     funcionCosto.m
%
%

%% INICIARLIZAR
clear ; close all; clc

%% Cargar datos -- analizar el archivo!!!!

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Parte 1: graficar ====================

fprintf(['Graficando los datos. \n']);

plotData(X, y);

hold on;
xlabel('Exámen 1 nota')
ylabel('Exámen 2 nota')

legend('Admitido', 'No admitido')
hold off;

fprintf('\nPrograma pausado.\n');
pause;


%% ============ Parte 2: Calcular costo y gradiente ============
%  funcionCosto.m

[m, n] = size(X);
X = [ones(m, 1) X];

% parámetros theta
initial_theta = zeros(n + 1, 1);

% Costo inicial
[cost, grad] = funcionCosto(initial_theta, X, y);

fprintf('Costo inicial (zeros): %f\n', cost);
fprintf('Gradiente con el costo inicial (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nPrograma pausado. \n');
pause;


%% ============= Parte 3: Optimizando con fminunc  =============


%  opciones
opciones = optimset('GradObj', 'on', 'MaxIter', 400);

%  fminunc
[theta, cost] = ...
	fminunc(@(t)(funcionCosto(t, X, y)), initial_theta, opciones);

% imprimir
fprintf('Costo según fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);


plotDecisionBoundary(theta, X, y);

% labels
hold on;
xlabel('Exámen 1 nota')
ylabel('Exámen 2 nota')

legend('Admitido', 'No admitido')
hold off;

fprintf('\nPrograma pausado. \n');
pause;

%% ============== Parte 4: Predicciones ==============
%
%  completar: predecir.m
%  Vamos a predecir la probabilidad de admisión de un estudiante cuyas notas fueron: exámen 1 -: 45 y exámen 2 -: 85

prob = sigmoid([1 45 85] * theta);
fprintf(['Para el estudiante con notas de 45 y 85, predecimos una probabilidad ' ...
         'de admisión de %f\n\n'], prob);

% Calcular la exactitud de nuestro modelo
p = predecir(theta, X);

fprintf('Exactitud de entrenamiento: %f\n', mean(double(p == y)) * 100);

fprintf('\nPrograma pausado. \n');
pause;
