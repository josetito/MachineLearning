function [theta, J_history] = gradienteDescendente(X, y, theta, alpha, num_iters)



m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== SU CÓDIGO ======================








    % ============================================================

    % guardar los valores del costo 
    J_history(iter) = costoMulti(X, y, theta);

end

end
