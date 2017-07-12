import numpy as np
import math

def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g