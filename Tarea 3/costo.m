function J = costo(X, y, theta)
%Costo de la regresión lineal
%   J = COSTO(X, y, theta) 

% Inicializar parámetros
m = length(y); % cantidad de examples

% el siguiente es el valor que se necesita retornar correctamente.
J = 0;

% ====================== SU CÓDIGO ======================

function hipotesis(x)
  theta(1) + theta(2) * x
end

promedio = 1/(2 * m)

hipotesis = X * theta;  %97*1   Bien
res = hipotesis .- y; % resta
res = res.^2; %elevamos al cuadrado

sumatoria = sum(res); %sumatoria
sumatoria = sumatoria * promedio; % resultado
sumatoria

% =========================================================================

end
