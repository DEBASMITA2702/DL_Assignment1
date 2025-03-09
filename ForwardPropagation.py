import numpy as np
import ActivationAndOutputFunctions
import Utilities

def forwardPropagation(weights, biases, input, num_total_layers, activation_function):
    pre_activations = {}
    activations = {}
    temp_bias = {}

    '''input activation is same as input'''
    activations[0] = input

    '''perform activation functions from input (which is layer 1) to pre output layer (which is layer total_layers - 2)'''
    for layer in range(1, num_total_layers - 1):
        temp_bias[layer] = Utilities.make_row_vector(biases[layer])
        temp_bias[layer] = Utilities.expand_by_repeating(temp_bias[layer], input.shape[1]).transpose()
        
        pre_activations[layer] = temp_bias[layer] + Utilities.multiply(weights[layer], activations[layer - 1])

        if activation_function == "sigmoid":
            activations[layer] = ActivationAndOutputFunctions.sigmoid(pre_activations[layer])
        elif activation_function == "tanh":
            activations[layer] = ActivationAndOutputFunctions.tanh(pre_activations[layer])
        elif activation_function == "relu":
            activations[layer] = ActivationAndOutputFunctions.relu(pre_activations[layer])
        else:
            activations[layer] = ActivationAndOutputFunctions.identity(pre_activations[layer])

    '''perform softmax fucntion in the output layer (which is layer total_layers - 1)'''
    temp_bias[num_total_layers - 1] = Utilities.make_row_vector(biases[num_total_layers - 1])
    temp_bias[num_total_layers - 1] = Utilities.expand_by_repeating(temp_bias[num_total_layers - 1], input.shape[1]).transpose()
    pre_activations[num_total_layers - 1] = temp_bias[num_total_layers - 1] + Utilities.multiply(weights[num_total_layers - 1], activations[num_total_layers - 2])
    temp = pre_activations[num_total_layers - 1].transpose()
    softmax_outputs = list()
    for current_class in range(len(temp)):
        softmax_value = ActivationAndOutputFunctions.softmax(temp[current_class])
        softmax_outputs.append(softmax_value)
    predicted_output = np.array(softmax_outputs)

    return pre_activations, activations, predicted_output.T


def forwardPropagationForPredictions(weights, biases, input, num_total_layers, activation_function):
    activations = {}
    pre_activations = {}
    
    '''input activation is same as input'''
    activations[0] = input

    '''perform activation functions from input to pre output layer'''
    for layer in range(1, num_total_layers - 1):
        pre_activations[layer] = biases[layer] + np.dot(weights[layer], activations[layer - 1])

        if activation_function == "sigmoid":
            activations[layer] = ActivationAndOutputFunctions.sigmoid(pre_activations[layer])
        elif activation_function == "tanh":
            activations[layer] = ActivationAndOutputFunctions.tanh(pre_activations[layer])
        elif activation_function == "relu":
            activations[layer] = ActivationAndOutputFunctions.relu(pre_activations[layer])
        else:
            activations[layer] = ActivationAndOutputFunctions.identity(pre_activations[layer])

    '''perform softmax fucntion in the output layer'''
    pre_activations[num_total_layers - 1] = biases[num_total_layers - 1] + np.dot(weights[num_total_layers - 1], activations[num_total_layers - 2])
    predicted_output = ActivationAndOutputFunctions.softmax(pre_activations[num_total_layers - 1])

    return pre_activations, activations, predicted_output