import numpy as np
import matplotlib.pyplot as plt
import sys

# Training data
X = np.array([
    [0,1],
    [1,0],
    [1,1],
    [0,0]
])

# Label of data
y = np.array([
    [1],
    [1],
    [0],
    [0]
])

# Learning rate for gradient descent
learning_rate = 0.01

# The parameter to help with overfitting
reg_param = 0

# Maximum iterations for gradient descent
max_iter = 5000

# Number of training examples
m = 4

## Set the typology 
input_units = 2 # Number of input units
hidden_units = 2 # Number of hidden units
output_units = 1 # Number of outout units

# Weights and biases
np.random.seed(1)
W1 = np.random.normal(0, 1, (hidden_units, input_units))
W2 = np.random.normal(0,1, (output_units, hidden_units))
B1 = np.random.random((hidden_units, 1))
B2 = np.random.random((output_units, 1))

print(W1)
print(W2)
print(B1)
print(B2)

def sigmoid(z, derv=False):
    if derv is True:
        return z * (1 - z)
    return 1 / (1 + np.exp(-z))


def train(W1, W2, B1, B2): # The arguments are to bypass UnboundLocalError error
    for i in range(max_iter):
        c = 0
        dW1 = 0
        dW2 = 0
        dB1 = 0
        dB2 = 0
        
        for j in range(m):
            sys.stdout.write("\rIteration: {} and {}".format(i + 1, j + 1))

            # Forward Prop.
            a0 = X[j].reshape(X[j].shape[0], 1) # 2x1
            print("Here is a0: %s" % (a0))

            z1 = W1.dot(a0) + B1 # 2x2 * 2x1 + 2x1 = 2x1
            a1 = sigmoid(z1) # 2x1
            print("Here is a1: %s" % (a1))
            print("Hello")

            z2 = W2.dot(a1) + B2 # 1x2 * 2x1 + 1x1 = 1x1
            a2 = sigmoid(z2) # 1x1

            # Back prop.
            dz2 = a2 - y[j] # 1x1
            print("Here is dz2 %s" % (dz2))
            dW2 += dz2 * a1.T # 1x1 .* 1x2 = 1x2
            print("Here is a1 %s" % (a1))

            dz1 = np.multiply((W2.T * dz2), sigmoid(a1, derv=True)) # (2x1 * 1x1) .* 2x1 = 2x1
            dW1 += dz1.dot(a0.T) # 2x1 * 1x2 = 2x2

            dB1 += dz1 # 2x1
            dB2 += dz2 # 1x1

            c = c + (-(y[j] * np.log(a2)) - ((1 - y[j]) * np.log(1 - a2)))
            sys.stdout.flush() # Updating the text.
        
        W1 = W1 - learning_rate * (dW1 / m) + ( (reg_param / m) * W1)
        W2 = W2 - learning_rate * (dW2 / m) + ( (reg_param / m) * W2)

        B1 = B1 - learning_rate * (dB1 / m)
        B2 = B2 - learning_rate * (dB2 / m)
    return (W1, W2, B1, B2)

W1, W2, B1, B2 = train(W1, W2, B1, B2)
