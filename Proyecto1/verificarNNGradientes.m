function verificarNNGradientes(lambda)

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);

X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];


costFunc = @(p) nnFuncionCosto(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);

[cost, grad] = costFunc(nn_params);
numgrad = gradientChecking(costFunc, nn_params);


disp([numgrad grad]);
fprintf(['Las columnas deben ser similares.\n' ...
         '(Izquierda-Su datos / Derecha-cálculo manual)\n\n']);


diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['Si la implementación de backpropagation es correcta, entonces \n' ...
         'la diferencia debe ser pequeña (menor a 1e-9). \n' ...
         '\nDiferencia: %g\n'], diff);

end
