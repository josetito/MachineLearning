function g = sigmoidGradient(z)
%SIGMOIDGRADIENT retorna el gradiente de la función sigmoide


g = zeros(size(z));

% ====================== CÓDIGO ======================

g = sigmoid(z).*(1 - sigmoid(z));


% =============================================================



end
