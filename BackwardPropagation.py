import numpy as np
import ActivationAndOutputFunctionsGradients
import Utilities

def backwardPropagation(weights, activations, pre_activations, trueOutput, predictedOutput, loss_function, num_total_layers, activation_function):
    derivative_pre_activations = {}
    derivative_weights = {}
    derivative_biases = {}
    derivative_activations = {}

    '''for output layer set values based on cross entropy or mse loss'''
    if loss_function == "cross_entropy":
        derivative_pre_activations[num_total_layers - 1] = -(trueOutput - predictedOutput)
    elif loss_function == "mean_squared_error":
        derivative_pre_activations[num_total_layers - 1] = (predictedOutput - trueOutput) * predictedOutput * (1 - predictedOutput)

    '''iterate till input layer'''
    for layer in range(num_total_layers - 1, 0 , -1):
        derivative_weights[layer] = Utilities.multiply(derivative_pre_activations[layer], activations[layer - 1].T)
        derivative_biases[layer] = np.sum(derivative_pre_activations[layer], axis = 1)
        derivative_activations[layer - 1] = Utilities.multiply(weights[layer].T, derivative_pre_activations[layer])

        if layer > 1:
            if activation_function == "sigmoid":
                derivative_pre_activations[layer - 1] = Utilities.matrixMultiply(derivative_activations[layer - 1], ActivationAndOutputFunctionsGradients.sigmoidGradient(pre_activations[layer - 1]))
            elif activation_function == "tanh":
                derivative_pre_activations[layer - 1] = Utilities.matrixMultiply(derivative_activations[layer - 1], ActivationAndOutputFunctionsGradients.tanhGradient(pre_activations[layer - 1]))
            elif activation_function == "relu":
                derivative_pre_activations[layer - 1] = Utilities.matrixMultiply(derivative_activations[layer - 1], ActivationAndOutputFunctionsGradients.reluGradient(pre_activations[layer - 1]))
            else:
                derivative_pre_activations[layer - 1] = Utilities.matrixMultiply(derivative_activations[layer - 1], ActivationAndOutputFunctionsGradients.identityGradient(pre_activations[layer - 1]))

    return derivative_weights, derivative_biases