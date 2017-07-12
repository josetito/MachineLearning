import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnFuncionCosto(Theta1,Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1):
    #Unrolling
    #Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1))

    #Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1))

    #variables
    m = len(X)

    # variables a retornar
    J = 0
    Theta1_grad = np.zeros((hidden_layer_size,(input_layer_size+1)))
    Theta2_grad = np.zeros((num_labels,hidden_layer_size+1))

    # ====================== CODIGO ======================


    I = np.eye(num_labels)
    Y = np.zeros((m, num_labels))

    #Y = np.zeros((num_labels, m))  #  10 x 60000
    for i in range(m):
        Y[i,] = I[y[i]]
    Y = Y

    A1 = np.column_stack((np.ones((m)),X)) #shape (6000,785)
    #print A1.shape
    #print Theta1.shape          #(25,785)

    transTheta1 = Theta1.T
    Z2= np.dot(A1,transTheta1)
    #print Z2.shape         # 60000 x 25
    #print Z2

    Z2 = np.array(Z2, dtype=np.float128)
    sigZ2 =sigmoid(Z2)
    #print "z2 sigmoide"
    #print sigZ2.shape      # 60000 x 25

    A2 = np.column_stack((np.ones((len(sigZ2))),sigZ2))
    #print "A2"
    #print A2.shape    #60000 x 26

    transTheta2 = Theta2.T
    Z3 = np.dot(A2,transTheta2)
    #print "Z3"   # shape 60000 x 10
    #print Z3

    Hipotesis = A3 = sigmoid(Z3)
    #print "hip"
    #print Hipotesis.shape    # (60000, 10)


    #formula de regularizacion
    # regularizacion = (lambda1/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)))
    Theta1Ala2 = Theta1[:,1:]**2   #quitando la primera columna
    Theta2Ala2 = Theta2[:,1:]**2
    #print "theta1"
    #print Theta1Ala2
    Theta1Suma = Theta1Ala2.sum()
    Theta2Suma = Theta2Ala2.sum()
    suma = Theta2Suma + Theta1Suma
    #print "suma"
    #print suma
    regularizacion = np.dot((lambda1/(2.0*m)),suma)
    print "regu"
    print regularizacion


    #Formula de costo
    #J = (1/m)*sum(sum((-Y).*log(Hipotesis) - (1-Y).*log(1-Hipotesis), 2))
    hip= np.log(Hipotesis)
    hip1= np.log(1-Hipotesis)
    multi = -Y * hip
    multi1= (1-Y) * hip1
    Resta = multi - multi1
    suma1 = Resta.sum()
    #print suma1
    J = (1.0/m) * suma1
    J = J + regularizacion

    print "j"
    print J



    Error3 = A3 - Y    #60000 x 10
    Error3 = np.array(Error3)
    print Error3.shape
    print Theta2.shape


    # Error2 = (Error3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);
    onesZ2 = np.column_stack((np.ones((len(Z2))), Z2))
    Error3Theta2 = np.dot(Error3,Theta2)
    sigGrad = sigmoidGradient(onesZ2)
    sigGrad = np.array(sigGrad)
    print sigGrad.shape #60000 x 26
    multiError2 = Error3Theta2 * sigGrad #60000x26
    print multiError2.shape #26x26
    Error2 = multiError2[:,1:]
    d = Error2.shape
    print d

    Delta1 = np.dot(Error2.T , A1)
    Delta2 = np.dot(Error3.T, A2)
    print Delta1.shape
    print Delta2.shape


    #Theta1_grad = Delta1/m + (lambda1/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)]
    #Theta2_grad = Delta2/m + (lambda1/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)]


    TempGrad = np.concatenate((np.zeros((len(Theta1),1)),Theta1[:,1:]), axis=1)
    #print tempGrad
    Theta1_grad = Delta1/m + (lambda1/m) * TempGrad
    print Theta1_grad.shape

    TempGrad2 = np.concatenate((np.zeros((len(Theta2), 1)), Theta2[:, 1:]), axis=1)
    Theta2_grad = Delta2/m + (lambda1/m) * TempGrad2
    print Theta2_grad.shape

    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradientes
    #grad = [Theta1_grad(:) ; Theta2_grad(:)]
    TempGrad3 = Theta1_grad.flatten(1)
    TempGrad4 = Theta2_grad.flatten(1)
    grad = np.concatenate((TempGrad3,TempGrad4),axis=0)
    print grad.shape
    print grad 

    return J
