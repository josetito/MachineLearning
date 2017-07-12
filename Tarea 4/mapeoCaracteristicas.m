function out = mapeoCaracteristicas(X1, X2)
%
%   mapea dos entradas en características cuadráticas (agregamos más)
%
%   Retura nuevas características:
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 son del mismo tamaño
%

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end
