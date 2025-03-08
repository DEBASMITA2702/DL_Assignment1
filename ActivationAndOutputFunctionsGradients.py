import numpy as np
from ActivationAndOutputFunctions import softmax, sigmoid
import Utilities

def sigmoidGradient(input):
    '''finding the sigmoid value of the input'''
    sigmoid_value = sigmoid(input)

    '''performing gradient of sigmoid'''
    gradient = sigmoid_value * (1 - sigmoid_value)

    return gradient

def tanhGradient(input):
    return 1-(np.tanh(input)**2)

    tanh_value = np.tanh(input)

    gradient = 1 - (tanh_value**2)

    return gradient

def reluGradient(input):
    return 1 * (input > 0)

def identityGradient(input):
    dim1 = input.shape[0]
    dim2 = input.shape[1]

    gradient = Utilities.create_ones(dim1, dim2)
    
    return gradient

def softmaxGradient(input):
    softmax_value = softmax(input)

    gradient = softmax_value * (1 - softmax_value)

    return gradient