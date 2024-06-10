# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:38:55 2024

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import math
from sc_fmt import *

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Computer Modern Roman"],'figure.figsize':  (7.0, 4.0)
})



#-----------------------
# MINI-BATCH RANDOMIZER
#-----------------------

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    inc = mini_batch_size

    # Step 2 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 64 examples.
    num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X =shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y =shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:
        
        mini_batch_X=shuffled_X[:,num_complete_minibatches*mini_batch_size:]
        mini_batch_Y=shuffled_Y[:,num_complete_minibatches*mini_batch_size:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



#-----------------------
# ACTIVATION FUNCTIONS
#-----------------------

def sigmoid(z):
    ''' 
    Sigmoid activation function.

    Inputs:
        z: numpy array, input to the activation function

    Outputs:
        result: numpy array, output after applying the sigmoid function
    '''
    return 1 / (1 + np.exp(-z))

def relu(z):
    ''' 
    ReLU activation function.

    Inputs:
        z: numpy array, input to the activation function

    Outputs:
        result: numpy array, output after applying the ReLU function
    '''
    return np.maximum(0, z)


#-------------------------------------
# DERIVATIVES OF ACTIVATION FUNCTIONS
#-------------------------------------

def diff_sigmoid(z):
    ''' 
    Derivative of the sigmoid function.

    Inputs:
        z: numpy array, input to the activation function

    Outputs:
        result: numpy array, derivative of the sigmoid function
    '''
    return sigmoid(z) * (1 - sigmoid(z))


def diff_relu(z):
    ''' 
    Derivative of the ReLU function.

    Inputs:
        z: numpy array, input to the activation function

    Outputs:
        result: numpy array, derivative of the ReLU function
    '''
    return np.heaviside(z, 0)

def diff_tanh(z):
    ''' 
    Derivative of the tanh function.

    Inputs:
        z: numpy array, input to the activation function

    Outputs:
        result: numpy array, derivative of the tanh function
    '''
    return 1 - np.tanh(z)**2


#------------------------------
# INITIALIZATION OF PARAMETERS
#------------------------------

def initialize_parameters(X, dims_layers):
    ''' 
    Initializes the parameters of the neural network.

    Inputs:
        X: numpy array, input features with shape (n_x, m)
        dims_layers: list, dimensions of each layer in the network

    Outputs:
        parameters: dictionary, containing initialized weights and biases
    '''
    np.random.seed(3)
    dims = np.concatenate(([X.shape[0]], dims_layers))
    parameters = {}
    for l in range(1, len(dims)):
        parameters['W' + str(l)] = np.random.randn(dims[l], dims[l-1]) * np.sqrt( 2 / dims[l-1] )
        parameters['b' + str(l)] = np.zeros((dims[l], 1))

        
        assert parameters['W' + str(l)].shape == (dims[l], dims[l-1])
        assert parameters['b' + str(l)].shape == (dims[l], 1)
        
    return parameters


def initialize_velocity(parameters):
    ''' 
    Initializes the velocity for momentum optimization.

    Inputs:
        parameters: dictionary, containing the weights and biases

    Outputs:
        V_grads: dictionary, initialized velocities
    '''
    L = len(parameters) // 2
    V_grads = {}
    for l in range(1, L + 1):
        V_grads["dW" + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        V_grads["db" + str(l)] = np.zeros(parameters['b' + str(l)].shape)
    return V_grads

def initialize_adam(parameters):
    ''' 
    Initializes the moment estimates for ADAM optimization.

    Inputs:
        parameters: dictionary, containing the weights and biases

    Outputs:
        V_grads: dictionary, initialized first moment estimates
        S_grads: dictionary, initialized second moment estimates
    '''
    L = len(parameters) // 2
    V_grads = {}
    S_grads = {}
    for l in range(1, L + 1):
        V_grads["dW" + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        V_grads["db" + str(l)] = np.zeros(parameters['b' + str(l)].shape)
        S_grads["dW" + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        S_grads["db" + str(l)] = np.zeros(parameters['b' + str(l)].shape)
    return V_grads, S_grads


#-----------------------
#  FORWARD PROPAGATION
#-----------------------

def forward_propagation_layer(X, W, b, activation='relu'):
    ''' 
    Computes forward propagation for a single layer.

    Inputs:
        X: numpy array, input data
        W: numpy array, weights
        b: numpy array, biases
        activation: string, type of activation function ('relu', 'sigmoid', 'tanh')

    Outputs:
        A: numpy array, output of the activation function
        Z: numpy array, linear transformation before activation
    '''
    Z = np.dot(W, X) + b
    assert Z.shape == (W.shape[0], X.shape[1])
    
    if activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'relu':
        A = relu(Z)
    elif activation == 'tanh':
        A = np.tanh(Z)
    
    return A, Z


def forward_propagation(X, parameters, activation='relu', keep_prob=1):
    ''' 
    Computes forward propagation through the entire network.

    Inputs:
        X: numpy array, input data
        parameters: dictionary, containing the weights and biases
        activation: string, type of activation function for hidden layers
        keep_prob: float, probability of keeping a neuron active during dropout

    Outputs:
        AL: numpy array, output of the last layer
        cache: dictionary, containing intermediate values needed for backpropagation
    '''
    L = len(parameters) // 2
    A = X
    cache = {}
    np.random.seed(1)
    
    for l in range(1, L):
        Wl = parameters['W' + str(l)]
        bl = parameters['b' + str(l)]
        A, Z = forward_propagation_layer(A, Wl, bl)
        
        if keep_prob < 1:
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D / keep_prob
            cache['D' + str(l)] = D
        
        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A

    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    AL, ZL = forward_propagation_layer(A, WL, bL, activation='sigmoid')
    cache['Z' + str(L)] = ZL
    cache['A' + str(L)] = AL
    
    return AL, cache


#-----------------------
#  BACKWARD PROPAGATION
#-----------------------

def back_propagation_layer(dZl, Wl, Zl_minus_1, Al_minus_1, lambd=0, activation='relu'):
    ''' 
    Computes backpropagation for a single layer.

    Inputs:
        dZl: numpy array, gradient of the cost with respect to Zl
        Wl: numpy array, weights
        Zl_minus_1: numpy array, linear transformation from the previous layer
        Al_minus_1: numpy array, activation from the previous layer
        lambd: float, regularization parameter
        activation: string, type of activation function

    Outputs:
        dZl_minus_1: numpy array, gradient of the cost with respect to Zl-1
        dWl: numpy array, gradient of the cost with respect to Wl
        dbl: numpy array, gradient of the cost with respect to bl
    '''
    if activation == 'relu':
        diffgl = diff_relu(Zl_minus_1)
    elif activation == 'sigmoid':
        diffgl = diff_sigmoid(Zl_minus_1)
    elif activation == 'tanh':
        diffgl = diff_tanh(Zl_minus_1)

    m = Al_minus_1.shape[1]
    dWl = (np.dot(dZl, Al_minus_1.T) + lambd * Wl) / m
    dbl = np.sum(dZl, axis=1, keepdims=True) / m
    dAl_minus_1 = np.dot(Wl.T, dZl)
    dZl_minus_1 = dAl_minus_1 * diffgl
    
    return dZl_minus_1, dWl, dbl


def back_propagation(X, Y, parameters, cache, lambd=0, activation='relu', keep_prob=1):
    ''' 
    Computes backpropagation through the entire network.

    Inputs:
        X: numpy array, input data
        Y: numpy array, true labels
        parameters: dictionary, containing the weights and biases
        cache: dictionary, containing intermediate values from forward propagation
Podrías continuar?
ChatGPT
python
Copiar código
        lambd: float, regularization parameter
        activation: string, type of activation function for hidden layers
        keep_prob: float, probability of keeping a neuron active during dropout

    Outputs:
        grads: dictionary, containing gradients with respect to the weights and biases
    '''
    L = len(parameters) // 2
    m = Y.shape[1]
    AL = cache['A' + str(L)]
    dZL = AL - Y
    grads = {'dZ' + str(L): dZL}
    
    dZl = dZL
    for l in range(L, 1, -1):
        Wl = parameters['W' + str(l)]
        Zl_minus_1 = cache['Z' + str(l-1)]
        Al_minus_1 = cache['A' + str(l-1)]
        
        dZl, dWl, dbl = back_propagation_layer(dZl, Wl, Zl_minus_1, Al_minus_1, lambd, activation)
        
        if keep_prob < 1:
            Dl = cache['D' + str(l-1)]
            dZl *= Dl/keep_prob
        
        grads['dW' + str(l)] = dWl
        grads['db' + str(l)] = dbl
        grads['dZ' + str(l-1)] = dZl
    
    dW1 = (np.dot(dZl, X.T) + lambd * parameters['W1']) / m
    db1 = np.sum(dZl, axis=1, keepdims=True) / m
    grads['dW1'] = dW1
    grads['db1'] = db1
    
    return grads

#-------------------------------------
#    COMPUTATION OF COST FUNCTION
#-------------------------------------

def compute_cost(AL, Y):
    ''' 
    Computes the cross-entropy cost.

    Inputs:
        AL: numpy array, probabilities from the last layer
        Y: numpy array, true labels

    Outputs:
        cost: float, cross-entropy cost
    '''
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = np.sum(logprobs)
    return np.squeeze(cost)

def compute_cost_with_regularization(AL, Y, parameters, lambd):
    ''' 
    Computes the cost with L2 regularization.

    Inputs:
        AL: numpy array, probabilities from the last layer
        Y: numpy array, true labels
        parameters: dictionary, containing the weights
        lambd: float, regularization parameter

    Outputs:
        cost: float, regularized cost
    '''
    cross_entropy_cost = compute_cost(AL, Y)
    L = len(parameters) // 2
    L2_regularization_cost = sum([np.sum(np.square(parameters['W' + str(l)])) for l in range(1, L + 1)]) * lambd / (2)
    return cross_entropy_cost + L2_regularization_cost


#-----------------------
#   UPDATE PARAMETERS
#-----------------------

def update_parameters(parameters, grads, learning_rate):
    ''' 
    Updates the parameters using gradient descent.

    Inputs:
        parameters: dictionary, containing the weights and biases
        grads: dictionary, containing gradients with respect to the weights and biases
        learning_rate: float, learning rate for gradient descent

    Outputs:
        parameters: dictionary, updated weights and biases
    '''
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters


def update_parameters_momentum(parameters, grads, V_grads, learning_rate, beta):
    ''' 
    Updates the parameters using momentum optimization.

    Inputs:
        parameters: dictionary, containing the weights and biases
        grads: dictionary, containing gradients with respect to the weights and biases
        V_grads: dictionary, velocities
        learning_rate: float, learning rate for gradient descent
        beta: float, momentum hyperparameter

    Outputs:
        parameters: dictionary, updated weights and biases
        V_grads: dictionary, updated velocities
    '''
    L = len(parameters) // 2
    for l in range(1, L + 1):
        V_grads['dW' + str(l)] = beta * V_grads['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        V_grads['db' + str(l)] = beta * V_grads['db' + str(l)] + (1 - beta) * grads['db' + str(l)]
        parameters['W' + str(l)] -= learning_rate * V_grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * V_grads['db' + str(l)]
    return parameters, V_grads


def update_parameters_adam(parameters, grads, V_grads, S_grads, idx_it, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    ''' 
    Updates the parameters using ADAM optimization.

    Inputs:
        parameters: dictionary, containing the weights and biases
        grads: dictionary, containing gradients with respect
Podrias continuar?
ChatGPT
python
Copiar código
        to the weights and biases
        V_grads: dictionary, velocities
        S_grads: dictionary, squared gradients
        idx_iteration: int, current iteration index
        learning_rate: float, learning rate for gradient descent
        beta1: float, exponential decay hyperparameter for the first moment estimates
        beta2: float, exponential decay hyperparameter for the second moment estimates
        epsilon: float, small value to prevent division by zero

    Outputs:
        parameters: dictionary, updated weights and biases
        V_grads: dictionary, updated velocities
        S_grads: dictionary, updated squared gradients
    '''
    L = len(parameters) // 2
    V_corrected = {}
    S_corrected = {}
    for l in range(1, L + 1):
        V_grads['dW' + str(l)] = beta1 * V_grads['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        V_grads['db' + str(l)] = beta1 * V_grads['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
        
        S_grads['dW' + str(l)] = beta2 * S_grads['dW' + str(l)] + (1 - beta2) * np.square(grads['dW' + str(l)])
        S_grads['db' + str(l)] = beta2 * S_grads['db' + str(l)] + (1 - beta2) * np.square(grads['db' + str(l)])
        
        V_corrected['dW' + str(l)] = V_grads['dW' + str(l)] / (1 - beta1 ** idx_it)
        V_corrected['db' + str(l)] = V_grads['db' + str(l)] / (1 - beta1 ** idx_it)
        
        S_corrected['dW' + str(l)] = S_grads['dW' + str(l)] / (1 - beta2 ** idx_it)
        S_corrected['db' + str(l)] = S_grads['db' + str(l)] / (1 - beta2 ** idx_it)
        
        parameters['W' + str(l)] -= learning_rate * V_corrected['dW' + str(l)] / (np.sqrt(S_corrected['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] -= learning_rate * V_corrected['db' + str(l)] / (np.sqrt(S_corrected['db' + str(l)]) + epsilon)
    return parameters, V_grads, S_grads


#-----------------------
#  LEARNING RATE DECAY
#-----------------------

def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer.
    decay_rate -- Decay rate. Scalar.
    time_interval -- Number of epochs where you update the learning rate.

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    # (approx. 1 lines)
    # learning_rate = ...
    # YOUR CODE STARTS HERE
    
    learning_rate=1/(1+decay_rate*int(epoch_num/time_interval)) *learning_rate0
    
    # YOUR CODE ENDS HERE
    return learning_rate



#-----------------------------------------------------------
#********************** MODEL TRAINING *********************
#-----------------------------------------------------------

def dnn_model(X, Y, dims_layers, learning_rate = 0.3, activation='relu', mini_batch_size = 64, 
              num_epochs=30000, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
              keep_prob = 1, optimizer = 'gd', lambd = 0,
              print_cost = False, print_iter = 10000, plot_cost = True, 
              decay=None, decay_rate=1):
    ''' 
    Trains a deep neural network.

    Inputs:
        X -- numpy array, input features
        Y -- numpy array, true labels
        dims_layers -- list, dimensions of each layer in the network
        learning_rate -- float, learning rate for gradient descent
        num_epochs -- int, number of iterations
        beta1 -- float, exponential decay hyperparameter for the first moment (ADAM)
        beta2 -- float, exponential decay hyperparameter for the second moment (ADAM)
        keep_prob -- float, probability of keeping a neuron active during dropout
        optimizer -- string, optimization method ('gd', 'momentum', 'adam')
        lambd -- float, regularization parameter
        print_cost -- boolean, whether to print the cost every 100 iterations
        print_it -- int, number of iterations to print the cost
        plot_cost -- boolean, whether to plot the cost variation over training

    Outputs:
        parameters -- dictionary, trained weights and biases
    '''
    
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]                   # number of training examples
    cost_history = []                # to keep track of the cost
    lr_rates = []
    learning_rate0 = learning_rate   # the original learning rate
    
    # Initialize parameters
    parameters = initialize_parameters(X, dims_layers)
    
    # Initialize the optimizer
    if optimizer != 'gd' :   
        # Initialize ADAM parameters if optimizer is 'adam', otherwise initialize velocity parameters
        V_grads, S_grads = initialize_adam(parameters) if optimizer == 'adam' else (initialize_velocity(parameters), None)

    # Optimization loop
    for i in tqdm(range(num_epochs), desc="Training progress"):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:
            
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
        
            # Forward propagation
            AL, cache = forward_propagation(minibatch_X, parameters, activation, keep_prob)
            
            # Backward propagation
            grads = back_propagation(minibatch_X, minibatch_Y, parameters, cache, lambd, activation, keep_prob)
            
            # Compute cost with or without regularization
            cost_total += compute_cost_with_regularization(AL, minibatch_Y, parameters, lambd) if lambd != 0 else compute_cost(AL, minibatch_Y)
                        
            # Update parameters based on the optimizer
            if optimizer == 'gd':
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, V_grads = update_parameters_momentum(parameters, grads, V_grads, learning_rate, beta1)
            elif optimizer == 'adam':
                t = t + 1 # Adam counter
                parameters, V_grads, S_grads = update_parameters_adam(parameters, grads, V_grads, S_grads, t, learning_rate, beta1, beta2, epsilon)
       
        cost = cost_total / m
        
        #Decay rate
        if decay:
            learning_rate = decay(learning_rate0, i, decay_rate)
        # Calculate cost every certain number of iterations
        if i % 100 == 0:
            cost_history.append(cost)
            if print_cost and i % print_iter == 0:
                print("Cost after iteration {}: {}".format(i, cost))
                if decay:
                    print("learning rate after epoch {}: {:.2e}".format(i, learning_rate))

    if plot_cost:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cost_history)
        ax.set_xlabel(r'$Epochs~(x{})$'.format(print_iter))
        ax.set_ylabel(r'$Cost:~J(\vec{W},\vec{b})$')
        
        #Title
        n=exponent(learning_rate)
        a=int_part(learning_rate)
        if n < -3:
            ax.set_title(r'Cost Variation over Training: $\alpha=%.1f \cdot{10}^{%i}$'%(a,n))
        else:
            ax.set_title(r'Cost Variation over Training: $\alpha={:.3f}$'.format(learning_rate))
            

    return parameters

#-----------------------
#      PREDICTIONS
#-----------------------

def predict(X, parameters, activation='relu'):
    ''' 
    Computes predictions using the trained neural network.

    Inputs:
        X: numpy array, input features
        parameters: dictionary, containing the weights and biases
        activation: string, type of activation function for hidden layers

    Outputs:
        Y_predict: numpy array, predicted labels
    '''
    AL, _ = forward_propagation(X, parameters, activation)
    Y_predict = AL.round()
    
    return Y_predict

#-----------------------
#        ACCURACY
#-----------------------

def accuracy(Y_predict, Y):
    ''' 
    Computes the accuracy of the predictions.

    Inputs:
        Y_predict: numpy array, predicted labels
        Y: numpy array, true labels

    Outputs:
        None (prints accuracy)
    '''
    acc = (Y - Y_predict).astype(int)
    correct = (acc == 0).sum()
    total = Y.shape[1]
    accuracy = correct * 100 / total
    
    print('Accuracy: ', accuracy, '%')
