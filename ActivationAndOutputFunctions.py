import numpy as np
import Utilities


def sigmoid(input):
    '''
      Parameters:
        input : input for which to find the sigmoid
      Returns:
        sigmoid value of the input
      Function:
        finds and returns the sigmoid value which acts as an activation function
    '''
    
    # clipping the value between -150 and 150 so that overflow does not occur when doing the exponent
    clipped_input = Utilities.clip_value(input, 150)

    # finding exponent
    exponent_clipped_input = Utilities.exponent(-clipped_input)

    # finding sigmoid value
    sigmoid_value = 1 / (1 + exponent_clipped_input)
    
    return sigmoid_value


def tanh(input):
    '''
      Parameters:
        input : input for which to find the tanh
      Returns:
        tanh value of the input
      Function:
        finds and returns the tanh value which acts as an activation function
    '''

    tanh_input = np.tanh(input)
    
    return tanh_input


def relu(input):
    '''
      Parameters:
        input : input for which to find the relu
      Returns:
        relu value of the input
      Function:
        finds and returns the relu value which acts as an activation function
    '''

    # if the input is greater than 0 then relu is the input itself. else it is 0
    relu_input = np.maximum(input, 0)
    
    return relu_input


def identity(input):
    '''
      Parameters:
        input : input for which to find the identity
      Returns:
        identity value of the input
      Function:
        finds and returns the identity value which acts as an activation function
    '''

    # identity function is the same as the input
    identity_input = input

    return identity_input


def softmax(input):
    '''
      Parameters:
        input : input for which to find the softmax
      Returns:
        softmax value of the input
      Function:
        finds and returns the softmax value which acts as an output function
    '''

    # finding the max value from the input vector
    max_input = np.max(input)

    #findind a normalized exponent so that overflow does not occur
    exponent_a = Utilities.exponent(input - max_input)

    # finding softmax (probability over a range of the output classes)
    sum_exp = np.sum(exponent_a)
    softmax_value = exponent_a / sum_exp

    return softmax_value