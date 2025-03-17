import numpy as np
from ForwardPropagation import forwardPropagation
from BackwardPropagation import backwardPropagation
import UpdateWeightsAndBiases

''' this file contains the different optimization algorithms'''

def stochastic_gradient_descent(NetworkModel, batch_size):
    '''
    Parameters:
        NetworkModel : an object of the model
        batch_size : batch size of the model
    Returns:
        weights : weights learned
        biases : biases learned
    Function:
        runs the stochastic gradient descent algorithm
    '''

    input_data_size = NetworkModel.x_train.shape[0]
    weights = NetworkModel.weights
    biases = NetworkModel.bias

    for batch in range(0, input_data_size, batch_size):
        input_batch = NetworkModel.x_train[batch : batch + batch_size]
        output_batch = NetworkModel.y_train[batch : batch + batch_size]

        # run forward propagation
        pre_activation, activation, predicted_output = forwardPropagation(weights, biases, input_batch.T, NetworkModel.num_total_layers, NetworkModel.activation_function)

        # run backward propagation
        weights_change, biases_change = backwardPropagation(weights, activation, pre_activation, output_batch.T, predicted_output, NetworkModel.loss_function, NetworkModel.num_total_layers, NetworkModel.activation_function)
        
        # apply update rule
        weights, biases = UpdateWeightsAndBiases.stochastic_update(weights, biases, NetworkModel.learning_rate, weights_change, biases_change, NetworkModel.weight_decay)
    
    return weights, biases


def momentum_based_gradient_descent(NetworkModel, history_weights, history_biases, beta, batch_size):
    '''
    Parameters:
        NetworkModel : an object of the model
        history_weights : history of the weights
        history_biases : history of the biases
        batch_size : batch size of the model
    Returns:
        weights : weights learned
        biases : biases learned
        history_weights : updated history of the weights
        history_biases : updated history of the biases
    Function:
        runs the momentum based gradient descent algorithm
    '''

    input_data_size = NetworkModel.x_train.shape[0]
    weights = NetworkModel.weights
    biases = NetworkModel.bias

    for batch in range(0, input_data_size, batch_size):
        input_batch = NetworkModel.x_train[batch : batch + batch_size]
        output_batch = NetworkModel.y_train[batch : batch + batch_size]

        # run forward propagation
        pre_activation, activation, predicted_output = forwardPropagation(weights, biases, input_batch.T, NetworkModel.num_total_layers, NetworkModel.activation_function)

        # run backward propagation
        weights_change, biases_change = backwardPropagation(weights, activation, pre_activation, output_batch.T, predicted_output, NetworkModel.loss_function, NetworkModel.num_total_layers, NetworkModel.activation_function)
        
        history_update_weight = {}
        history_update_bias = {}
        for point in range(1, len(weights_change)):
          history_update_weight[point] = beta * history_weights[point] + NetworkModel.learning_rate * weights_change[point]
          history_update_bias[point] = beta * history_biases[point] + NetworkModel.learning_rate * biases_change[point]

        # apply update rule
        weights, biases = UpdateWeightsAndBiases.momentum_update(weights, biases, history_update_weight, history_update_bias, NetworkModel.learning_rate, NetworkModel.weight_decay)

        history_weights = history_update_weight
        history_biases = history_update_bias

    return  weights, biases, history_weights, history_biases


def nesterov_gradient_descent(NetworkModel, uw, ub, history_weights, history_biases, inital_weights, inital_biases, beta, batch_size):
    '''
    Parameters:
        NetworkModel : an object of the model
        uw : weight for nesterov
        ub : bias for nesterov
        history_weights : history of the weights
        history_biases : history of the biases
        inital_weights : initial weights for the epoch
        inital_biases : initial bias for the epoch
        batch_size : batch size of the model
    Returns:
        weights : weights learned
        biases : biases learned
        history_weights : updated history of the weights
        history_biases : updated history of the biases
    Function:
        runs the nesterov accelerated gradient descent algorithm
    '''
    
    input_data_size = NetworkModel.x_train.shape[0]
    weights = NetworkModel.weights
    biases = NetworkModel.bias

    for batch in range(0, input_data_size, batch_size):
        original_weights = {}
        original_biases = {}

        input_batch = NetworkModel.x_train[batch : batch + batch_size]
        output_batch = NetworkModel.y_train[batch : batch + batch_size]

        # run forward propagation
        pre_activation, activation, predicted_output = forwardPropagation(inital_weights, inital_biases, input_batch.T, NetworkModel.num_total_layers, NetworkModel.activation_function)

        for layer in range(1, len(weights) + 1):
            original_weights[layer] = weights[layer]
            weights[layer] = inital_weights[layer]
            original_biases[layer] = biases[layer]
            biases[layer] = inital_biases[layer]

        # run backward propagation
        weights_change, biases_change = backwardPropagation(weights, activation, pre_activation, output_batch.T, predicted_output, NetworkModel.loss_function, NetworkModel.num_total_layers, NetworkModel.activation_function)

        for layer in range(1, len(weights) + 1):
            weights[layer] = original_weights[layer]
            biases[layer] = original_biases[layer]

        for layer in range(1, len(weights_change) + 1):
            uw[layer] = beta * history_weights[layer] + NetworkModel.learning_rate * weights_change[layer]
            ub[layer] = beta * history_biases[layer] + NetworkModel.learning_rate * biases_change[layer]

        # apply update rule
        weights, biases = UpdateWeightsAndBiases.momentum_update(weights, biases, uw, ub, NetworkModel.learning_rate, NetworkModel.weight_decay)

        history_weights = uw
        history_biases = ub

    return  weights, biases, history_weights, history_biases


def rmsprop_gradient_descent(NetworkModel, history_weights, history_biases, beta, epsilon, batch_size):
    '''
    Parameters:
        NetworkModel : an object of the model
        history_weights : history of the weights
        history_biases : history of the biases
        beta : beta value for the model
        epsilon : epsilon value for the model
        batch_size : batch size of the model
    Returns:
        weights : weights learned
        biases : biases learned
        history_weights : updated history of the weights
        history_biases : updated history of the biases
    Function:
        runs the rms prop gradient descent algorithm
    '''

    input_data_size = NetworkModel.x_train.shape[0]
    weights = NetworkModel.weights
    biases = NetworkModel.bias

    for batch in range(0, input_data_size, batch_size):
        input_batch = NetworkModel.x_train[batch : batch + batch_size]
        output_batch = NetworkModel.y_train[batch : batch + batch_size]
        
        # run forward propagation
        pre_activation, activation, predicted_output = forwardPropagation(weights, biases, input_batch.T, NetworkModel.num_total_layers, NetworkModel.activation_function)
        
        # run backward propagation
        weights_change, biases_change = backwardPropagation(weights, activation, pre_activation, output_batch.T, predicted_output, NetworkModel.loss_function, NetworkModel.num_total_layers, NetworkModel.activation_function)

        for layer in range(1, len(weights_change) + 1):
            history_weights[layer] = (beta * history_weights[layer]) + ((1 - beta) * (np.square(weights_change[layer])))
            history_biases[layer] = (beta * history_biases[layer]) + ((1 - beta) * (np.square(biases_change[layer])))

        # apply update rule
        weights, biases = UpdateWeightsAndBiases.rmsprop_update(weights, biases, NetworkModel.learning_rate, history_weights, history_biases, weights_change, biases_change, epsilon, NetworkModel.weight_decay)

    return  weights, biases, history_weights, history_biases


def adam_gradient_descent(NetworkModel, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat, beta1, beta2, epsilon, batch_size, epoch):
    '''
    Parameters:
        NetworkModel : an object of the model
        vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat : parameters for adam
        beta1 : beta1 value for the model
        beta2 : beta2 value for the model
        epsilon : epsilon value for the model
        batch_size : batch size of the model
        epoch : current epoch number
    Returns:
        weights : weights learned
        biases : biases learned
        vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat : updated parameters for the next epoch
    Function:
        runs the adam gradient descent algorithm
    '''

    input_data_size = NetworkModel.x_train.shape[0]
    weights = NetworkModel.weights
    biases = NetworkModel.bias

    for batch in range(0, input_data_size, batch_size):
        input_batch = NetworkModel.x_train[batch : batch + batch_size]
        output_batch = NetworkModel.y_train[batch : batch + batch_size]

        # run forward propagation
        pre_activation, activation, predicted_output = forwardPropagation(weights, biases, input_batch.T, NetworkModel.num_total_layers, NetworkModel.activation_function)

        # run backward propagation
        weights_change, biases_change = backwardPropagation(weights, activation, pre_activation, output_batch.T, predicted_output, NetworkModel.loss_function, NetworkModel.num_total_layers, NetworkModel.activation_function)

        for layer in range(1, len(weights_change) + 1):
            mw[layer] = (beta1 * mw[layer]) + ((1 - beta1) * weights_change[layer])
            mb[layer] = (beta1 * mb[layer]) + ((1 - beta1) * biases_change[layer])
            vw[layer] = (beta2 * vw[layer]) + ((1 - beta2) * (np.square(weights_change[layer])))
            vb[layer] = (beta2 * vb[layer]) + ((1 - beta2) * (np.square(biases_change[layer])))

        for layer in range(1, len(weights_change) + 1):
            mw_hat[layer] = mw[layer] / (1 - np.power(beta1, epoch + 1))
            mb_hat[layer] = mb[layer] / (1 - np.power(beta1, epoch + 1))
            vw_hat[layer] = vw[layer] / (1 - np.power(beta2, epoch + 1))
            vb_hat[layer] = vb[layer] / (1 - np.power(beta2, epoch + 1))

        # apply update rule
        weights, biases = UpdateWeightsAndBiases.adam_update(weights, biases, NetworkModel.learning_rate, mw_hat, mb_hat, vw_hat, vb_hat, epsilon, NetworkModel.weight_decay)

    return weights, biases, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat


def nadam_gradient_descent(NetworkModel, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat, beta1, beta2, epsilon, batch_size):
    '''
    Parameters:
        NetworkModel : an object of the model
        vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat : parameters for nadam
        beta1 : beta1 value for the model
        beta2 : beta2 value for the model
        epsilon : epsilon value for the model
        batch_size : batch size of the model
    Returns:
        weights : weights learned
        biases : biases learned
        vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat : updated parameters for the next epoch
    Function:
        runs the nadam gradient descent algorithm
    '''
    
    input_data_size = NetworkModel.x_train.shape[0]
    weights = NetworkModel.weights
    biases = NetworkModel.bias

    for batch in range(0, input_data_size, batch_size):
        input_batch = NetworkModel.x_train[batch : batch + batch_size]
        output_batch = NetworkModel.y_train[batch : batch + batch_size]

        # run forward propagation
        pre_activation, activation, predicted_output = forwardPropagation(weights, biases, input_batch.T, NetworkModel.num_total_layers, NetworkModel.activation_function)

        # run backward propagation
        weights_change, biases_change = backwardPropagation(weights, activation, pre_activation, output_batch.T, predicted_output, NetworkModel.loss_function, NetworkModel.num_total_layers, NetworkModel.activation_function)

        for layer in range(1, len(weights_change) + 1):
            mw[layer] = (beta1 * mw[layer]) + ((1 - beta1) * weights_change[layer])
            mb[layer] = (beta1 * mb[layer]) + ((1 - beta1) * biases_change[layer])
            vw[layer] = (beta2 * vw[layer]) + ((1 - beta2) * (np.square(weights_change[layer])))
            vb[layer] = (beta2 * vb[layer]) + ((1 - beta2) * (np.square(biases_change[layer])))

        for layer in range(1, len(weights_change) + 1):
            mw_hat[layer] = mw[layer] / (1 - np.power(beta1, layer + 1))
            mb_hat[layer] = mb[layer] / (1 - np.power(beta1, layer + 1))
            vw_hat[layer] = vw[layer] / (1 - np.power(beta2, layer + 1))
            vb_hat[layer] = vb[layer] / (1 - np.power(beta2, layer + 1))

        # apply update rule
        weights, biases = UpdateWeightsAndBiases.nadam_update(weights, biases, NetworkModel.learning_rate, mw_hat, mb_hat, vw_hat, vb_hat, beta1, beta2, weights_change, biases_change, epsilon, NetworkModel.weight_decay)

    return weights, biases, vw, vb, mw, mb, mw_hat, mb_hat, vw_hat, vb_hat