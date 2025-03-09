import numpy as np
import Utilities

def sigmoid(input):
    '''clipping the value netween -150 and 150 so that overflow does not occur'''
    clipped_input = Utilities.clip_value(input, 150)

    '''finding exponent'''
    exponent_clipped_input = Utilities.exponent(-clipped_input)

    '''performing sigmoid'''
    sigmoid_value = 1 / (1 + exponent_clipped_input)
    
    return sigmoid_value

def tanh(input):
    tanh_input = np.tanh(input)
    
    return tanh_input

def relu(input):
    relu_input = np.maximum(input, 0)
    
    return relu_input

def identity(input):
    identity_input = input

    return identity_input

def softmax(input):
    '''normalizing the value so that overflow does not occur'''
    max_input = np.max(input)
    exponent_a = Utilities.exponent(input - max_input)

    '''performing softmax'''
    sum_exp = np.sum(exponent_a)
    softmax_value = exponent_a / sum_exp

    return softmax_value