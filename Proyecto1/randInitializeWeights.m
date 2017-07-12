function W = randInitializeWeights(L_in, L_out)


% variable a retornar
W = zeros(L_out, 1 + L_in);
%l_out filas
%L_in columnas

% ====================== CÓDIGO ======================


epsilon_init = 0.2;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
% ========================================================================

end
