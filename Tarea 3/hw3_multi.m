%% ================ Sección 1: Feature Normalization ================


clear ; close all; clc

fprintf('Cargando datos ...\n');

data = load('hw3data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% imprimir algunos puntos
fprintf('Primeras 10 muestras de nuestro data set: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Pausa, presiones enter.\n');
pause;

% 
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% agregar columna
X = [ones(m, 1) X];


%% ================ Sección 2: Gradiente descendente ================


fprintf('Running gradient descent ...\n');

% elija un alpha
alpha = 0.01;
num_iters = 400;

% theta inicial y correr gradiente descendente
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% graficar
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% desplegar resultado
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimar el precio de una casa de 1650 pies cuadrados y 3 cuartos
% ====================== SU CÓDIGO ======================
% recordar que nuestros datos están normalizados
price = 0; % hay que cambiar esto


% ============================================================

fprintf(['Casa -  1650 pies cuadrados, 3 cuartos ' ...
         '(usando gradiente descendente):\n $%f\n'], price);%modificar código para poder predecir

fprintf('Pausa.\n');
pause;

%% ================ Sección 3: Ecuación normal ================

fprintf('Usando la ecuación normal...\n');

%% cargando Datos
data = csvread('hw3data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

X = [ones(m, 1) X];

% Calcular los parámetros
theta = ecuNormal(X, y);

% resultado
fprintf('Theta según la ecuación normal: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimar el precio de una casa de 1650 pies cuadrados y 3 cuartos
% ====================== SU CÓDIGO ======================
price = 0; % cambiar esto


% ============================================================

fprintf(['Casa -  1650 pies cuadrados, 3 cuartos ' ...
         '(usando ecuación normal):\n $%f\n'], price);%modificar código para poder predecir

