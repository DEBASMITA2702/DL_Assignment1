import numpy as np
import ActivationAndOutputFunctionsGradients
import Utilities


''' this function is used to run the backpropagation algorithm for the framework over a given input'''
def backwardPropagation(weights, activations, pre_activations, trueOutput, predictedOutput, loss_function, num_total_layers, activation_function):
    '''
      Parameters:
        weights : the weights which are used in the forward propagation
        activations : values of the activation layers
        pre_activations : values of the pre-activation layers
        trueOutput : true output for the input in form of one hot vectors (which is used to calculate derivate w.r.t to the output layer)
        predictedOutput : predicted output for the input from the output function (which is used to calculate derivate w.r.t to the output layer)
        loss_function : which loss function is used to calculate the loss
        num_total_layers : total number of layers in the framework (which is used to calculate the derivate w.r.t hidden layers)
        activation_function : which activation function is used in the framework
      Returns:
        derivative_weights : a dictionary storing the gradient (the change which should be adjusted with the weights to minimze the loss) of the weights across the different layers
        derivative_biases : a dictionary storing the gradient (the change which should be adjusted with the biases to minimze the loss) of the biases across the different layers
      Function:
        finds and returns the change in the weights and biases that should be done to minimize the loss
        this is calculated by finding the derivative of the loss w.r.t to all the parameters and layers involved in the framework
    '''

    # dictionaries to store the derivatives across different layers
    derivative_pre_activations = {}
    derivative_weights = {}
    derivative_biases = {}
    derivative_activations = {}

    # derivative w.r.t to the output layer (here the derivative depends on the loss function that is being used to calculate the loss)
    if loss_function == "cross_entropy":
        derivative_pre_activations[num_total_layers - 1] = -(trueOutput - predictedOutput)
    elif loss_function == "mean_squared_error":
        derivative_pre_activations[num_total_layers - 1] = (predictedOutput - trueOutput) * predictedOutput * (1 - predictedOutput)

    # derivative w.r.t the hidden and input layers
    for layer in range(num_total_layers - 1, 0 , -1):
        # finding the derivatives for weights, biases and activations
        derivative_weights[layer] = Utilities.multiply(derivative_pre_activations[layer], activations[layer - 1].T)
        derivative_biases[layer] = np.sum(derivative_pre_activations[layer], axis = 1)
        derivative_activations[layer - 1] = Utilities.multiply(weights[layer].T, derivative_pre_activations[layer])

        # for hidden layers there will be a derivative for the pre activation also (for input layer there is no pre activation layer)
        if layer > 1:
            # the respective gradient funtion is called based on the choice of the activation function
            if activation_function == "sigmoid":
                derivative_pre_activations[layer - 1] = Utilities.matrixMultiply(derivative_activations[layer - 1], ActivationAndOutputFunctionsGradients.sigmoidGradient(pre_activations[layer - 1]))
            elif activation_function == "tanh":
                derivative_pre_activations[layer - 1] = Utilities.matrixMultiply(derivative_activations[layer - 1], ActivationAndOutputFunctionsGradients.tanhGradient(pre_activations[layer - 1]))
            elif activation_function == "relu":
                derivative_pre_activations[layer - 1] = Utilities.matrixMultiply(derivative_activations[layer - 1], ActivationAndOutputFunctionsGradients.reluGradient(pre_activations[layer - 1]))
            else:
                derivative_pre_activations[layer - 1] = Utilities.matrixMultiply(derivative_activations[layer - 1], ActivationAndOutputFunctionsGradients.identityGradient(pre_activations[layer - 1]))

    # return the changes in weights and biases that are calculated
    return derivative_weights, derivative_biases