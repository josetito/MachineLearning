function plotData(X, y)

% CREAR NUEVA FIGURA
figure; hold on;

% ====================== CÓDIGO ======================
% Instrucciones: Graficar las muestras negativas y positivas en un gráfico 2D
%               usando la opción 'k+' para los positivos
%               y 'ko' para las muestras negativas.
%


plot(X, y, 'k+', 'MarkerSize', 10);



plot(X, y, 'ko', 'MarkerSize', 10);




% =========================================================================



hold off;

end
