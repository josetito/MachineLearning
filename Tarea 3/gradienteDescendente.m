function [theta, J_history] = gradienteDescendente(X, y, theta, alpha, num_iters)


% inicialización de algunos valores importantes
m = length(y); % número de training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== SU CÓDIGO ======================
    % Instrucciones: Ejecute actualización de parámetros
    %
    % Nota: Para debuggear podría ir imprimiendo el costo, con los parámetros encontrados en cada iteración
    %







    % ============================================================

    % Guardar el costo J de cada iteración
    % -esto no es necesario, pero lo usaremos para graficar más adelante-
    J_history(iter) = computeCost(X, y, theta);

end

end
