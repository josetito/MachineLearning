%% inicialización
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Parte 1: visualizar =============
%

% cargar
fprintf('Cargando y visualizando datos ...\n')

load('data1.mat');
m = size(X, 1);

% elegir aleatoriamente 100 imágenes
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Programa pausado.\n');
pause;


%% ================ Parte 2: cargar parámetros ================

fprintf('\nCargando parámetros guardados para la Red Neuronal...\n')

% Cargar variables en las variables Theta1 y Theta2
load('pesos.mat');

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Parte 3: Calcular Costo (Feedforward) ================
%  Primero vamos a implementar feedforward para calcular el costo solamente.
%
%  Primero hay que implementar el costo sin regularización!!
%
fprintf('\nFeedforward usando Redes Neuronales...\n')

% para regularización
lambda = 0;

J = nnFuncionCosto(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Costo inicial (parámetros cargados de pesos.mat): %f '...
         '\n(este valor debe ser cercano a 0.287629)\n'], J);

fprintf('\nPrograma pausado.\n');
pause;

%% =============== Parte 4: reglarización ===============
%

fprintf('\nVerificando la función de costo (con Regularización) ... \n')

lambda = 1;

J = nnFuncionCosto(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Costo de parámetros (cargados de pesos.mat): %f '...
         '\n(este valor debe ser cercano a 0.383770)\n'], J);

fprintf('Programa pausado.\n');
pause;


%% ================ Parte 5: Sigmoid Gradient  ================
%  Primero debe completar el código de sigmoidGradient.main
% esto porque necesitamos la derivada de la función sigmoide!!

fprintf('\nEvaluando sigmoid gradient...\n')

g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient para los valores [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Programa pausado.\n');
pause;


%% ================ Parte 6: inicializando p-arámetros ================

fprintf('\nInitializando parámetros para la Red Neuronal...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== PartE 7: Implementando Backpropagation ===============
%  Hay que seguir completando el código de nnFuncionCosto.m
%
fprintf('\nChequeando Backpropagation... \n');

%  Check gradients by running checkNNGradients
verificarNNGradientes;

fprintf('\nPrograma pausado.\n');
pause;


%% =============== Parte 8: Implementando Regularización ===============

fprintf('\nChequeando Backpropagation (con Regularización) ... \n')

lambda = 3;
verificarNNGradientes(lambda);

debug_J  = nnFuncionCosto(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCosto actual (con nuevo lambda): %f ' ...
         '\n(Este valor debe ser cercano a 0.576051)\n\n'], debug_J);

fprintf('Programa pausado.\n');
pause;


%% =================== Parte 8: Training  ===================
%
fprintf('\nEntrenando la Red Neuronal... \n')


options = optimset('MaxIter', 50);

%  se pueden cambiar los valores de lambda
lambda = 1;

% hacemos un shorthand
funcionCosto = @(p) nnFuncionCosto(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);


[nn_params, cost] = fmincg(funcionCosto, initial_nn_params, options);

% Obtener Theta1 y Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Programa pausado.\n');
pause;


%% ================= Parte 9: Visualizando pesos =================

fprintf('\nVisualizando pesos... \n')

displayData(Theta1(:, 2:end));

fprintf('\nPrograma pausado.\n');
pause;

%% ================= Parte 10: predecir =================

pred = predecir(Theta1, Theta2, X);

fprintf('\nPrecisión de la Red Neuronal sobre el training set: %f\n', mean(double(pred == y)) * 100);
