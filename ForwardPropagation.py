import numpy as np
import ActivationAndOutputFunctions
import Utilities

''' this function is used to run the forward propagation algorithm for the framework'''
def forwardPropagation(weights, biases, input, num_total_layers, activation_function):
    '''
    Parameters:
        weights : weight parameters
        biases : bias parameters
        input : input data
        num_total_layers : total number of layers for the model
        activation_function : activation function for the model
    Returns:
        pre_activations : values for the pre activation layers
        activations : values for the activation layers
        predicted_output : predicted output from the softmax function
    Function:
        runs the forward propagation algorithm
    '''
    
    # dictionaries to store the values of the pre activation and activation layers
    pre_activations = {}
    activations = {}
    temp_bias = {}

    # input layer activation is input itself
    activations[0] = input

    # apply the activation function from input (which is layer 1) to pre output layer (which is layer total_layers - 2)
    for layer in range(1, num_total_layers - 1):
        # calculate the values for the pre activation layer by using the weights and biases
        temp_bias[layer] = Utilities.make_row_vector(biases[layer])
        temp_bias[layer] = Utilities.expand_by_repeating(temp_bias[layer], input.shape[1]).transpose()
        
        pre_activations[layer] = temp_bias[layer] + Utilities.multiply(weights[layer], activations[layer - 1])

        # calculate the values for the activation layer by applying the activation function on the pre activation layer
        if activation_function == "sigmoid":
            activations[layer] = ActivationAndOutputFunctions.sigmoid(pre_activations[layer])
        elif activation_function == "tanh":
            activations[layer] = ActivationAndOutputFunctions.tanh(pre_activations[layer])
        elif activation_function == "relu":
            activations[layer] = ActivationAndOutputFunctions.relu(pre_activations[layer])
        else:
            activations[layer] = ActivationAndOutputFunctions.identity(pre_activations[layer])

    # apply softmax function in the output layer (which is layer total_layers - 1) to find the predicted output
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


''' this function is used to run the forward propagation algorithm on test data'''
def forwardPropagationForPredictions(weights, biases, input, num_total_layers, activation_function):
    '''
    Parameters:
        weights : weight parameters
        biases : bias parameters
        input : input data
        num_total_layers : total number of layers for the model
        activation_function : activation function for the model
    Returns:
        pre_activations : values for the pre activation layers
        activations : values for the activation layers
        predicted_output : predicted output from the softmax function
    Function:
        runs the forward propagation algorithm for the purpose of testing (i.e. input is test data)
    '''

    # dictionaries to store the values of the pre activation and activation layers
    activations = {}
    pre_activations = {}
    
    # input layer activation is input itself
    activations[0] = input

    # apply the activation function from input (which is layer 1) to pre output layer (which is layer total_layers - 2)
    for layer in range(1, num_total_layers - 1):
        # calculate the values for the pre activation layer by using the weights and biases
        pre_activations[layer] = biases[layer] + np.dot(weights[layer], activations[layer - 1])

        # calculate the values for the activation layer by applying the activation function on the pre activation layer
        if activation_function == "sigmoid":
            activations[layer] = ActivationAndOutputFunctions.sigmoid(pre_activations[layer])
        elif activation_function == "tanh":
            activations[layer] = ActivationAndOutputFunctions.tanh(pre_activations[layer])
        elif activation_function == "relu":
            activations[layer] = ActivationAndOutputFunctions.relu(pre_activations[layer])
        else:
            activations[layer] = ActivationAndOutputFunctions.identity(pre_activations[layer])

    # apply softmax function in the output layer (which is layer total_layers - 1) to find the predicted output
    pre_activations[num_total_layers - 1] = biases[num_total_layers - 1] + np.dot(weights[num_total_layers - 1], activations[num_total_layers - 2])
    predicted_output = ActivationAndOutputFunctions.softmax(pre_activations[num_total_layers - 1])

    return pre_activations, activations, predicted_output