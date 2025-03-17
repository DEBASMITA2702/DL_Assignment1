import numpy as np

''' this file conatains the update rules for the different optimization algorithms'''

def stochastic_update(weights, biases, learning_rate, weights_change, biases_change, weight_decay):
    '''
    Parameters:
        weights : weight parameters
        biases : bias parameters
        learning_rate : learning rate of the model
        weights_change : change in weights (learnt from backprop)
        biases_change : change in biases (learnt from backprop)
        weight_decay : weight decay of the model
    Returns:
        weights : updated weights
        biases : updated biases
    Function:
        runs the update rule for stochastic gradient descent algorithm
    '''

    for layer in range(1, len(weights_change) + 1):
        weights[layer] -= (learning_rate * weights_change[layer]) + (learning_rate * weight_decay * weights[layer])
    
    for layer in range(1, len(biases_change) + 1):
        biases[layer] -= (learning_rate * biases_change[layer])

    return weights, biases


def momentum_update(weights, biases, history_update_weight, history_update_bias, learning_rate, weight_decay):
    '''
    Parameters:
        weights : weight parameters
        biases : bias parameters
        learning_rate : learning rate of the model
        history_update_weight : history of weights (learnt from backprop)
        history_update_bias : history of biases (learnt from backprop)
        weight_decay : weight decay of the model
    Returns:
        weights : updated weights
        biases : updated biases
    Function:
        runs the update rule for momentum based and nesterov accelerated gradient descent algorithm
    '''
    for layer in range(1, len(history_update_weight) + 1):
        weights[layer] -= history_update_weight[layer] + (learning_rate * weight_decay * weights[layer])
    
    for layer in range(1, len(history_update_bias) + 1):
        biases[layer] -= history_update_bias[layer]
    
    return weights,biases


def rmsprop_update(weights, biases, learning_rate, history_weights, history_biases, weights_change, biases_change, epsilon, weight_decay):
    '''
    Parameters:
        weights : weight parameters
        biases : bias parameters
        learning_rate : learning rate of the model
        history_weights : history of weights
        history_biases : history of biases
        weights_change : change in weights (learnt from backprop)
        biases_change : change in biases (learnt from backprop)
        epsilon : epsilon value of the model
        weight_decay : weight decay of the model
    Returns:
        weights : updated weights
        biases : updated biases
    Function:
        runs the update rule for rms prop gradient descent algorithm
    '''

    for layer in range(1, len(history_weights) + 1):
        learning_rate_update_factor = np.sqrt(np.sum(history_weights[layer])) + epsilon
        updated_learning_rate = learning_rate / learning_rate_update_factor
        weights[layer] -= (updated_learning_rate * weights_change[layer]) + (learning_rate * weight_decay * weights[layer])

    for layer in range(1, len(history_biases) + 1):
        learning_rate_update_factor = np.sqrt(np.sum(history_biases[layer])) + epsilon
        updated_learning_rate = learning_rate / learning_rate_update_factor
        biases[layer] -= (updated_learning_rate * biases_change[layer])

    return weights, biases


def adam_update(weights, biases, learning_rate, mw_hat, mb_hat, vw_hat, vb_hat, epsilon, weight_decay):
    '''
    Parameters:
        weights : weight parameters
        biases : bias parameters
        learning_rate : learning rate of the model
        mw_hat, mb_hat, vw_hat, vb_hat : parameters for adam
        epsilon : epsilon value of the model
        weight_decay : weight decay of the model
    Returns:
        weights : updated weights
        biases : updated biases
    Function:
        runs the update rule for adam gradient descent algorithm
    '''

    for layer in range(1,len(vw_hat)+1):
        learning_rate_update_factor = np.sqrt(vw_hat[layer]) + epsilon
        updated_learning_rate = learning_rate / learning_rate_update_factor
        weights[layer] -= (updated_learning_rate * mw_hat[layer]) + (learning_rate * weight_decay * weights[layer])

    for layer in range(1,len(vb_hat)+1):
        learning_rate_update_factor = np.sqrt(vb_hat[layer]) + epsilon
        updated_learning_rate = learning_rate / learning_rate_update_factor
        biases[layer] -= (updated_learning_rate * mb_hat[layer])
    
    return weights, biases


def nadam_update(weights, biases, learning_rate, mw_hat, mb_hat, vw_hat, vb_hat, beta1, beta2, weights_change, biases_change, epsilon, weight_decay):
    '''
    Parameters:
        weights : weight parameters
        biases : bias parameters
        learning_rate : learning rate of the model
        mw_hat, mb_hat, vw_hat, vb_hat : parameters for adam
        beta1 : beta1 value for the model
        beta2 : beta2 value for the model
        weights_change : change in weights (learnt from backprop)
        biases_change : change in biases (learnt from backprop)
        epsilon : epsilon value of the model
        weight_decay : weight decay of the model
    Returns:
        weights : updated weights
        biases : updated biases
    Function:
        runs the update rule for adam gradient descent algorithm
    '''
    
    for layer in range(1, len(vw_hat) + 1):
        learning_rate_update_factor = np.sqrt(vw_hat[layer] + epsilon)
        updated_learning_rate = learning_rate/learning_rate_update_factor
        beta_factor = beta1 * mw_hat[layer]
        one_minus_beta_factor = ((1 - beta1) * weights_change[layer]) / (1 - beta1**(layer + 1))
        total_beta_factor = beta_factor + one_minus_beta_factor
        weights[layer] -= (updated_learning_rate * total_beta_factor) + (learning_rate * weight_decay * weights[layer])
    
    for layer in range(1, len(vb_hat) + 1):
        learning_rate_update_factor = np.sqrt(vb_hat[layer] + epsilon)
        updated_learning_rate = learning_rate / learning_rate_update_factor
        beta_factor = beta1 * mb_hat[layer]
        one_minus_beta_factor = ((1 - beta1) * biases_change[layer]) / (1 - beta1**(layer + 1))
        total_beta_factor = beta_factor + one_minus_beta_factor
        biases[layer] -= (updated_learning_rate * total_beta_factor)

    return weights, biases