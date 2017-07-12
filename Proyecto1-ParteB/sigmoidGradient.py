import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(z):
#SIGMOIDGRADIENT retorna el gradiente de la funcion sigmoide
    g = np.zeros((z.shape))

    g = sigmoid(z)*(1 - sigmoid(z))
    return g