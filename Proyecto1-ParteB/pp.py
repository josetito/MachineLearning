import random
import os
import struct
import numpy as np
from nnFuncionCosto import nnFuncionCosto



# Setup the parameters you will use for this exercise
input_layer_size  = 784;  # 28x28 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          #10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

# =========== Parte 1: visualizar =============

#cargar
print ('Cargando y visualizando datos ...\n')


from mnist import MNIST
mndata = MNIST('samples')
X, y = mndata.load_training()
m = len(X)

#print y  #60000
#print m  #60000

# elegir aleatoriamente 100 imagenes
#sel = randperm(size(X, 1));
#sel = sel(1:100);

#displayData(X(sel, :));

#print ('Programa pausado.\n');
#pause;


# ================ Parte 2: cargar parametros ================

print ('\nCargando parametros guardados para la Red Neuronal...\n')

#Theta1 = hidden_layer_size * 785
#Theta2 = 10 * (hidden_layer_size+1)

Theta1 = np.random.uniform(-1.0,1.0,hidden_layer_size*(input_layer_size+1))
Theta2 = np.random.uniform(-1.0,1.0,num_labels*(hidden_layer_size+1))

Theta1= Theta1.reshape(( hidden_layer_size, input_layer_size+1))
Theta2= Theta2.reshape((num_labels,hidden_layer_size+1))



#print [Theta1,Theta2]
#nn_params = np.concatenate((Theta1,Theta2), axis=0)



# Unroll parameters
#nn_params = [Theta1 ; Theta2]



#================ Parte 3: Calcular Costo (Feedforward) ================
#  Primero vamos a implementar feedforward para calcular el costo solamente.

#  Primero hay que implementar el costo sin regularizacion!!

print ('\nFeedforward usando Redes Neuronales...\n')

#para regularizacion
lambda1 = 0


J = nnFuncionCosto(Theta1,Theta2, input_layer_size, hidden_layer_size,num_labels, X, y, lambda1)

'''
print (['Costo inicial (parametros cargados de pesos.mat): %f '...
         '\n(este valor debe ser cercano a 0.287629)\n'], J);

print ('\nPrograma pausado.\n');
#pause;

# =============== Parte 4: reglarizacion ===============

print ('\nVerificando la funcion de costo (con Regularizacion) ... \n')

lambda = 1;

J = nnFuncionCosto(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

print (['Costo de parametros (cargados de pesos.mat): %f '...
         '\n(este valor debe ser cercano a 0.383770)\n'], J);

print ('Programa pausado.\n');
pause;


# ================ Parte 5: Sigmoid Gradient  ================
#  Primero debe completar el codigo de sigmoidGradient.main
# esto porque necesitamos la derivada de la funcion sigmoide!!

print ('\nEvaluando sigmoid gradient...\n')

g = sigmoidGradient([1 -0.5 0 0.5 1]);
print ('Sigmoid gradient para los valores [1 -0.5 0 0.5 1]:\n  ');
print ('%f ', g);
print ('\n\n');

print ('Programa pausado.\n');
pause;


# ================ Parte 6: inicializando p-arametros ================

print ('\nInitializando parametros para la Red Neuronal...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

# Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


# =============== PartE 7: Implementando Backpropagation ===============
#  Hay que seguir completando el codigo de nnFuncionCosto.m

print ('\nChequeando Backpropagation... \n');

# Check gradients by running checkNNGradients
verificarNNGradientes;

print ('\nPrograma pausado.\n');
pause;


# =============== Parte 8: Implementando Regularizacion ===============

print ('\nChequeando Backpropagation (con Regularizacion) ... \n')

lambda = 3;
verificarNNGradientes(lambda);

debug_J  = nnFuncionCosto(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

print (['\n\nCosto actual (con nuevo lambda): %f ' ...
         '\n(Este valor debe ser cercano a 0.576051)\n\n'], debug_J);

print ('Programa pausado.\n');
pause;


# =================== Parte 8: Training  ===================

print ('\nEntrenando la Red Neuronal... \n')


options = optimset('MaxIter', 50);

#  se pueden cambiar los valores de lambda
lambda = 1;

# hacemos un shorthand
funcionCosto = @(p) nnFuncionCosto(p, input_layer_size,hidden_layer_size, num_labels, X, y, lambda);


[nn_params, cost] = fmincg(funcionCosto, initial_nn_params, options);

# Obtener Theta1 y Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

print ('Programa pausado.\n');
pause;


#================= Parte 9: Visualizando pesos =================

print ('\nVisualizando pesos... \n')

displayData(Theta1(:, 2:end));

print ('\nPrograma pausado.\n');
pause;

# ================= Parte 10: predecir =================

pred = predecir(Theta1, Theta2, X);

print ('\nPrecision de la Red Neuronal sobre el training set: %f\n', mean(double(pred == y)) * 100);

'''