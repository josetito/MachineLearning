function [J grad] = nnFuncionCosto(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%Unrolling
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% variables
m = size(X, 1);   %5000

% variables a retornar
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== CÓDIGO ======================

Y = zeros(m,num_labels);
for i=1:num_labels
  Y(:,i) = (y == i);
end

A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';  %5000x401  401x25
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2 * Theta2';
Hipotesis = A3 = sigmoid(Z3);


regularizacion = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
J = (1/m)*sum(sum((-Y).*log(Hipotesis) - (1-Y).*log(1-Hipotesis), 2));
J = J + regularizacion;   % 50x1

Error3 = A3 - Y;
Error2 = (Error3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);
Delta1 = Error2'*A1;
Delta2 = Error3'*A2;


Theta1_grad = Delta1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];  %con regularizacion
Theta2_grad = Delta2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradientes
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
